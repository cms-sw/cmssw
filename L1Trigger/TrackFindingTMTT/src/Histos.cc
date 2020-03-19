#include "L1Trigger/TrackFindingTMTT/interface/Histos.h"
#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/Get3Dtracks.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrkRZfilter.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrk4and5.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TH2Poly.h>
#include <TF1.h>
#include <TPad.h>
#include <TProfile.h>
#include <TGraphAsymmErrors.h>
#include <TGraph.h>
#include <TEfficiency.h>

#include <algorithm>
#include <array>
#include <unordered_set>

using namespace std;

namespace TMTT {

//=== Store cfg parameters.

Histos::Histos(const Settings* settings) : settings_(settings), plotFirst_(true), bApproxMistake_(false) {
  genMinStubLayers_ = settings->genMinStubLayers();
  numPhiSectors_    = settings->numPhiSectors();
  numEtaRegions_    = settings->numEtaRegions();
  houghMinPt_       = settings->houghMinPt();
  houghNbinsPt_     = settings->houghNbinsPt();
  houghNbinsPhi_    = settings->houghNbinsPhi();
  chosenRofZ_       = settings->chosenRofZ();
  trackFitters_     = settings->trackFitters();
  useRZfilter_      = settings->useRZfilter();
  ranRZfilter_      = (useRZfilter_.size() > 0); // Was any r-z track filter run?
  resPlotOpt_       = settings->resPlotOpt(); // Only use signal events for helix resolution plots?
}

//=== Book all histograms

void Histos::book() {
  // Don't bother booking histograms if user didn't request them via TFileService in their cfg.
  if ( ! this->enabled() ) return;

  TH1::SetDefaultSumw2(true);

  // Book histograms about input data.
  this->bookInputData();
  // Book histograms checking if (eta,phi) sector definition choices are good.
  this->bookEtaPhiSectors();
  // Book histograms checking filling of r-phi HT array.
  this->bookRphiHT();
  // Book histograms about r-z track filters.
  if (ranRZfilter_) this->bookRZfilters();
  // Book histograms for studying freak, extra large events at HT.
  this->bookStudyBusyEvents();
  // Book histograms studying 3D track candidates found after HT.
  this->bookTrackCands("HT");
  // Book histograms studying 3D track candidates found after r-z track filter.
  if (ranRZfilter_) this->bookTrackCands("RZ");
  // Book histograms studying track fitting performance
  this->bookTrackFitting();
}

//=== Fill all histograms

void Histos::fill(const InputData& inputData, const matrix<Sector>& mSectors, const matrix<HTrphi>& mHtRphis, 
    	          const matrix<Get3Dtracks> mGet3Dtrks, const std::map<std::string,std::vector<L1fittedTrack>>& fittedTracks) 
{
  // Don't bother filling histograms if user didn't request them via TFileService in their cfg.
  if ( ! this->enabled() ) return;

  // Fill histograms about input data.
  this->fillInputData(inputData);
  // Fill histograms checking if (eta,phi) sector definition choices are good.
  this->fillEtaPhiSectors(inputData, mSectors);
  // Fill histograms checking filling of r-phi HT array.
  this->fillRphiHT(mHtRphis);
  // Fill histograms about r-z track filters.
  if (ranRZfilter_) this->fillRZfilters(mGet3Dtrks);
  // Fill histograms for studying freak, extra large events at HT.
  this->fillStudyBusyEvents(inputData, mSectors, mHtRphis, mGet3Dtrks);
  // Fill histograms studying 3D track candidates found after HT.
  vector<L1track3D> tracksHT;
  bool withRZfilter = false; 
  for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
    const Get3Dtracks& get3Dtrk = mGet3Dtrks(iPhiSec, iEtaReg);
    const std::vector< L1track3D >& tracks = get3Dtrk.trackCands3D(withRZfilter);
    tracksHT.insert(tracksHT.end(), tracks.begin(), tracks.end());
  }
  this->fillTrackCands(inputData, tracksHT, "HT");
  // Fill histograms studying 3D track candidates found after r-z track filter.
  if (ranRZfilter_) {
    vector<L1track3D> tracksRZ;
    bool withRZfilter = true; 
    for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
      const Get3Dtracks& get3Dtrk = mGet3Dtrks(iPhiSec, iEtaReg);
      const std::vector< L1track3D >& tracks = get3Dtrk.trackCands3D(withRZfilter);
      tracksRZ.insert(tracksRZ.end(), tracks.begin(), tracks.end());
    }
    this->fillTrackCands(inputData, tracksRZ, "RZ");
  }
  // Fill histograms studying track fitting performance
  this->fillTrackFitting(inputData, fittedTracks);
}

//=== Book histograms using input stubs and tracking particles.

TFileDirectory Histos::bookInputData() {
  TFileDirectory inputDir = fs_->mkdir("InputData");

  // N.B. Histograms of the kinematics and production vertex of tracking particles
  // are booked in bookTrackCands(), since they are used to study the tracking efficiency.

  hisNumEvents_ = inputDir.make<TH1F>("NumEvents",";; No. of events",1,-0.5,0.5);

  // Count stubs & tracking particles.

  profNumStubs_        = inputDir.make<TProfile>("NumStubs","; Category; No. stubs in tracker",4,0.5,4.5);
  profNumStubs_->GetXaxis()->SetBinLabel(1,"All stubs");
  profNumStubs_->GetXaxis()->SetBinLabel(2,"Genuine stubs");
  profNumStubs_->GetXaxis()->SetBinLabel(3,"Stubs matched to TP");
  profNumStubs_->GetXaxis()->SetBinLabel(4,"Stubs matched to TP for eff");
  profNumStubs_->LabelsOption("d");

  hisStubsVsEta_      = inputDir.make<TH1F>("StubsVsEta","; #eta; No. stubs in tracker",30,-3.0,3.0);
  hisStubsVsR_        = inputDir.make<TH1F>("StubsVsR","; radius (cm); No. stubs in tracker",1200,0.,120.);

  hisStubsVsRVsZ_      = inputDir.make<TH2F>("StubsVsRVsZ","; z (cm); radius (cm); No. stubs in tracker",1000,-280,280,1000,0,130);
  hisStubsModuleVsRVsZ_      = inputDir.make<TH2F>("StubsModuleVsRVsZ","; z (cm); radius (cm); No. stubs in tracker",1000,-280,280,1000,0,130);
  hisStubsVsRVsPhi_      = inputDir.make<TH2F>("StubsVsRVsPhi","; x (cm); y (cm); No. stubs in tracker",1000,-130,130,1000,-130,130);
  hisStubsModuleVsRVsPhi_      = inputDir.make<TH2F>("StubsModuleVsRVsPhi","; x (cm); y (cm); No. stubs in tracker",1000,-130,130,1000,-130,130);
  hisStubsModuleTiltVsZ_      = inputDir.make<TH2F>("StubsModuleTiltVsZ","; z (cm); Tilt; Module tilt vs z",1000,-280,280,128,-3.2,3.2);
  hisStubsdPhiCorrectionVsZ_      = inputDir.make<TH2F>("StubsdPhiCorrectionVsZ","; z (cm); Correction; dPhi Correction vs z",1000,-280,280,100,-1,10);

  hisStubsVsRVsZ_outerModuleAtSmallerR_      = inputDir.make<TH2F>("StubsVsRVsZ_outerModuleAtSmallerR","; z (cm); radius (cm); No. stubs in tracker",1000,-280,280,1000,0,130);
  hisStubsVsRVsPhi_outerModuleAtSmallerR_    = inputDir.make<TH2F>("StubsVsRVsPhi_outerModuleAtSmallerR","; x (cm); y (cm); No. stubs in tracker",1000,-130,130,1000,-130,130);

  profNumTPs_          = inputDir.make<TProfile>("NumTPs","; Category; No. of TPs in tracker",3,0.5,3.5);
  profNumTPs_->GetXaxis()->SetBinLabel(1,"All TPs");
  profNumTPs_->GetXaxis()->SetBinLabel(2,"TPs for eff.");
  profNumTPs_->GetXaxis()->SetBinLabel(3,"TPs for alg. eff.");
  profNumTPs_->LabelsOption("d");

  hisNumStubsPerTP_    = inputDir.make<TH1F>("NumStubsPerTP","; Number of stubs per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSStubsPerTP_    = inputDir.make<TH1F>("NumPSStubsPerTP","; Number of PS stubs per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SStubsPerTP_    = inputDir.make<TH1F>("Num2SStubsPerTP","; Number of 2S stubs per TP for alg. eff.",50,-0.5,49.5);

  hisNumLayersPerTP_    = inputDir.make<TH1F>("NumLayersPerTP","; Number of layers per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSLayersPerTP_    = inputDir.make<TH1F>("NumPSLayersPerTP","; Number of PS layers per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SLayersPerTP_    = inputDir.make<TH1F>("Num2SLayersPerTP","; Number of 2S layers per TP for alg. eff.",50,-0.5,49.5);

  hisNumLayersPerTP_muons_    = inputDir.make<TH1F>("NumLayersPerTP_muons","; Number of layers per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSLayersPerTP_muons_    = inputDir.make<TH1F>("NumPSLayersPerTP_muons","; Number of PS layers per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SLayersPerTP_muons_    = inputDir.make<TH1F>("Num2SLayersPerTP_muons","; Number of 2S layers per TP for alg. eff.",50,-0.5,49.5);

  hisNumLayersPerTP_electrons_    = inputDir.make<TH1F>("NumLayersPerTP_electrons","; Number of layers per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSLayersPerTP_electrons_    = inputDir.make<TH1F>("NumPSLayersPerTP_electrons","; Number of PS layers per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SLayersPerTP_electrons_    = inputDir.make<TH1F>("Num2SLayersPerTP_electrons","; Number of 2S layers per TP for alg. eff.",50,-0.5,49.5);

  hisNumLayersPerTP_pions_    = inputDir.make<TH1F>("NumLayersPerTP_pions","; Number of layers per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSLayersPerTP_pions_    = inputDir.make<TH1F>("NumPSLayersPerTP_pions","; Number of PS layers per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SLayersPerTP_pions_    = inputDir.make<TH1F>("Num2SLayersPerTP_pions","; Number of 2S layers per TP for alg. eff.",50,-0.5,49.5);

  hisNumLayersPerTP_lowPt_    = inputDir.make<TH1F>("NumLayersPerTP_lowPt","; Number of layers per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSLayersPerTP_lowPt_    = inputDir.make<TH1F>("NumPSLayersPerTP_lowPt","; Number of PS layers per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SLayersPerTP_lowPt_    = inputDir.make<TH1F>("Num2SLayersPerTP_lowPt","; Number of 2S layers per TP for alg. eff.",50,-0.5,49.5);

  hisNumLayersPerTP_mediumPt_    = inputDir.make<TH1F>("NumLayersPerTP_mediumPt","; Number of layers per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSLayersPerTP_mediumPt_    = inputDir.make<TH1F>("NumPSLayersPerTP_mediumPt","; Number of PS layers per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SLayersPerTP_mediumPt_    = inputDir.make<TH1F>("Num2SLayersPerTP_mediumPt","; Number of 2S layers per TP for alg. eff.",50,-0.5,49.5);

  hisNumLayersPerTP_highPt_    = inputDir.make<TH1F>("NumLayersPerTP_highPt","; Number of layers per TP for alg. eff.",50,-0.5,49.5);
  hisNumPSLayersPerTP_highPt_    = inputDir.make<TH1F>("NumPSLayersPerTP_highPt","; Number of PS layers per TP for alg. eff.",50,-0.5,49.5);
  hisNum2SLayersPerTP_highPt_    = inputDir.make<TH1F>("Num2SLayersPerTP_highPt","; Number of 2S layers per TP for alg. eff.",50,-0.5,49.5);

  // Study efficiency of tightened front end-electronics cuts.

  hisStubKillFE_          = inputDir.make<TProfile>("StubKillFE","; barrelLayer or 10+endcapRing; Stub fraction rejected by readout chip",30,-0.5,29.5);
  hisStubIneffiVsInvPt_   = inputDir.make<TProfile>("StubIneffiVsPt","; 1/Pt; Inefficiency of readout chip for good stubs",30,0.0,1.0);
  hisStubIneffiVsEta_     = inputDir.make<TProfile>("StubIneffiVsEta","; |#eta|; Inefficiency of readout chip for good stubs",30,0.0,3.0);
  hisStubKillDegradeBend_ = inputDir.make<TProfile>("StubKillDegradeBend","; barrelLayer or 10+endcapRing; Stub fraction killed by DegradeBend.h window cut",30,-0.5,29.5);

  // Study stub resolution.

  hisPtStub_             = inputDir.make<TH1F>("PtStub","; Stub q/Pt",50,-0.5,0.5);
  hisPtResStub_          = inputDir.make<TH1F>("PtResStub","; Stub q/Pt minus TP q/Pt",50,-0.5,0.5);
  hisBendFilterPower_    = inputDir.make<TH1F>("BendFilterPower","; Fraction of q/Pt range allowed",102,-0.01,1.01);
  hisDelPhiStub_         = inputDir.make<TH1F>("DelPhiStub","; Stub bend angle",50,-0.2,0.2);
  hisDelPhiResStub_      = inputDir.make<TH1F>("DelPhiResStub","; Stub bend angle minus TP bend angle",200,-0.2,0.2);

  hisDelPhiResStub_tilted_      = inputDir.make<TH1F>("DelPhiResStub_tilted","; Stub bend angle minus TP bend angle",200,-0.2,0.2);
  hisDelPhiResStub_notTilted_      = inputDir.make<TH1F>("DelPhiResStub_notTilted","; Stub bend angle minus TP bend angle",200,-0.2,0.2);

  hisBendStub_           = inputDir.make<TH1F>("BendStub","; Stub bend in units of strips",57,-7.125,7.125);
  hisBendResStub_        = inputDir.make<TH1F>("BendResStub","; Stub bend minus TP bend in units of strips",100,-5.,5.);
  hisNumMergedBend_      = inputDir.make<TH1F>("NumMergedBend","; No. of bend values merged together by loss of bit",10,-0.5,9.5);
  hisBendVsLayerOrRingPS_  = inputDir.make<TH2F>("BendVsLayerOrRingPS","; PS barrelLayer or 10+endcapRing; Stub bend",30,-0.5,29.5,57,-7.125,7.125);
  hisBendVsLayerOrRing2S_  = inputDir.make<TH2F>("BendVsLayerOrRing2S","; 2S barrelLayer or 10+endcapRing; Stub bend",30,-0.5,29.5,57,-7.125,7.125);
  hisBendFEVsLayerOrRingPS_  = inputDir.make<TH2F>("BendFEVsLayerOrRingPS","; PS barrelLayer or 10+endcapRing; Stub bend in FE chip",30,-0.5,29.5,57,-7.125,7.125);
  hisBendFEVsLayerOrRing2S_  = inputDir.make<TH2F>("BendFEVsLayerOrRing2S","; 2S barrelLayer or 10+endcapRing; Stub bend in FE chip",30,-0.5,29.5,57,-7.125,7.125);

  hisPhiStubVsPhiTP_     = inputDir.make<TH1F>("PhiStubVsPhiTP","; Stub #phi minus TP #phi at stub radius",100,-0.05,0.05);
  hisPhiStubVsPhi0TP_    = inputDir.make<TH1F>("PhiStubVsPhi0TP","; Stub #phi minus TP #phi0",100,-0.3,0.3);
  hisPhi0StubVsPhi0TP_   = inputDir.make<TH1F>("Phi0StubVsPhi0TP","; #phi0 of Stub minus TP",100,-0.2,0.2);
  hisPhi0StubVsPhi0TPres_= inputDir.make<TH1F>("Phi0StubVsPhi0TPres","; #phi0 of Stub minus TP / resolution",100,-5.0,5.0);
  hisPhiStubVsPhi65TP_   = inputDir.make<TH1F>("PhiStubVsPhi65TP","; Stub #phi minus TP phitrk65",100,-0.2,0.2);
  hisPhi65StubVsPhi65TP_ = inputDir.make<TH1F>("Phi65StubVsPhi65TP","; phitrk65 of Stub minus TP",100,-0.2,0.2);
  hisPhi65StubVsPhi65TPres_ = inputDir.make<TH1F>("Phi65StubVsPhi65TPres","; phitrk65 of Stub minus TP / resolution",100,-5.0,5.0);

  // Note ratio of sensor pitch to separation (needed to understand how many bits this can be packed into).
  hisPitchOverSep_       = inputDir.make<TH1F>("PitchOverSep","; ratio of sensor pitch / separation",100,0.0,0.1);
  hisRhoParameter_       = inputDir.make<TH1F>("RhoParameter","; rho parameter",100,0.0,0.2);
  // Check alpha correction.
  hisAlphaCheck_         = inputDir.make<TH2F>("AlphaCheck", "; #phi from stub; #phi from strip",40,-0.2,0.2,40,-0.2,0.2);
  // Count stubs sharing a common cluster.
  hisFracStubsSharingClus0_ = inputDir.make<TH1F>("FracStubsSharingClus0","Fraction of stubs sharing cluster in seed sensor",102,-0.01,1.01);
  hisFracStubsSharingClus1_ = inputDir.make<TH1F>("FracStubsSharingClus1","Fraction of stubs sharing cluster in correlation sensor",102,-0.01,1.01);

  hisStubB_ = inputDir.make<TH1F>("StubB","Variable B for all stubs on TP",100,0.9,10);
  hisStubBApproxDiff_tilted_ = inputDir.make<TH1F>("StubBApproxDiff_tilted_","Difference between exact and approximate values for B",100,-1,1);

  // Histos for denominator of tracking efficiency 
  float maxAbsQoverPt = 1. / houghMinPt_; // Max. |q/Pt| covered by  HT array.
  unsigned int nPhi   = numPhiSectors_;
  unsigned int nEta   = numEtaRegions_;
  hisTPinvptForEff_ = inputDir.make<TH1F>("TPinvptForEff", "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisTPptForEff_    = inputDir.make<TH1F>("TPptForEff", "; Pt of TP (used for effi. measurement);",25,0.0,100.0);
  hisTPetaForEff_   = inputDir.make<TH1F>("TPetaForEff","; #eta of TP (used for effi. measurement);",20,-3.,3.);
  hisTPphiForEff_   = inputDir.make<TH1F>("TPphiForEff","; #phi of TP (used for effi. measurement);",20,-M_PI,M_PI);
  hisTPd0ForEff_    = inputDir.make<TH1F>("TPd0ForEff", "; d0 of TP (used for effi. measurement);",40,0.,4.);
  hisTPz0ForEff_    = inputDir.make<TH1F>("TPz0ForEff", "; z0 of TP (used for effi. measurement);",50,0.,25.);
  //
  hisTPinvptForAlgEff_   = inputDir.make<TH1F>("TPinvptForAlgEff", "; 1/Pt of TP (used for alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisTPptForAlgEff_      = inputDir.make<TH1F>("TPptForAlgEff", "; Pt of TP (used for alg. effi. measurement);",25,0.0,100.0);
  hisTPetaForAlgEff_     = inputDir.make<TH1F>("TPetaForAlgEff","; #eta of TP (used for alg. effi. measurement);",20,-3.,3.);
  hisTPphiForAlgEff_     = inputDir.make<TH1F>("TPphiForAlgEff","; #phi of TP (used for alg. effi. measurement);",20,-M_PI,M_PI);
  hisTPd0ForAlgEff_      = inputDir.make<TH1F>("TPd0ForAlgEff", "; d0 of TP (used for alg. effi. measurement);",40,0.,4.);
  hisTPz0ForAlgEff_      = inputDir.make<TH1F>("TPz0ForAlgEff", "; z0 of TP (used for alg. effi. measurement);",50,0.,25.);
  //
  hisTPphisecForAlgEff_  = inputDir.make<TH1F>("TPphisecForAlgEff", "; #phi sectorof TP (used for alg. effi. measurement);",nPhi,-0.5,nPhi-0.5);
  hisTPetasecForAlgEff_  = inputDir.make<TH1F>("TPetasecForAlgEff", "; #eta sector of TP (used for alg. effi. measurement);",nEta,-0.5,nEta-0.5);
  //
  hisTPinvptForAlgEff_inJetPtG30_ = inputDir.make<TH1F>("TPinvptForAlgEff_inJetPtG30", "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisTPinvptForAlgEff_inJetPtG100_ = inputDir.make<TH1F>("TPinvptForAlgEff_inJetPtG100", "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisTPinvptForAlgEff_inJetPtG200_ = inputDir.make<TH1F>("TPinvptForAlgEff_inJetPtG200", "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);

  return inputDir;
}

//=== Fill histograms using input stubs and tracking particles.

void Histos::fillInputData(const InputData& inputData) {
  const vector<const Stub*>& vStubs = inputData.getStubs();
  const vector<TP>&          vTPs   = inputData.getTPs();

  hisNumEvents_->Fill(0.);

  // Count stubs.
  unsigned int nStubsGenuine = 0;
  unsigned int nStubsWithTP = 0;
  unsigned int nStubsWithTPforEff = 0;
  for (const Stub* stub : vStubs) {
    if (stub->genuine()) {
      nStubsGenuine++;
      if (stub->assocTP() != nullptr) {
        nStubsWithTP++;
        if (stub->assocTP()->useForEff()) nStubsWithTPforEff++;
      }
    }
  }
  profNumStubs_->Fill(1, vStubs.size());
  profNumStubs_->Fill(2, nStubsGenuine);
  profNumStubs_->Fill(3, nStubsWithTP);
  profNumStubs_->Fill(4, nStubsWithTPforEff);

  for (const Stub* stub : vStubs) {
    hisStubsVsEta_->Fill(stub->eta());
    hisStubsVsR_->Fill(stub->r());
    hisStubsVsRVsZ_->Fill( stub->z(), stub->r() );
    hisStubsModuleVsRVsZ_->Fill( stub->minZ(), stub->minR() );
    hisStubsModuleVsRVsZ_->Fill( stub->maxZ(), stub->maxR() );

    hisStubsModuleTiltVsZ_->Fill( stub->minZ(), stub->moduleTilt() );
    hisStubsModuleTiltVsZ_->Fill( stub->maxZ(), stub->moduleTilt() );

    if ( stub->barrel() && stub->outerModuleAtSmallerR() ) {
      hisStubsVsRVsZ_outerModuleAtSmallerR_->Fill( stub->z(), stub->r() );
    }

    hisStubsdPhiCorrectionVsZ_->Fill(  stub->minZ(), stub->dphiOverBendCorrection() );

    hisStubsVsRVsPhi_->Fill( stub->r() * sin( stub->phi() ), stub->r() * cos( stub->phi() ) );
    hisStubsModuleVsRVsPhi_->Fill( stub->minR() * sin( stub->minPhi() ), stub->minR() * cos( stub->minPhi() ) );
    hisStubsModuleVsRVsPhi_->Fill( stub->maxR() * sin( stub->maxPhi() ), stub->maxR() * cos( stub->maxPhi() ) );

    if ( stub->barrel() && stub->outerModuleAtSmallerR() ) {
      hisStubsVsRVsPhi_outerModuleAtSmallerR_->Fill( stub->r() * sin( stub->phi() ), stub->r() * cos( stub->phi() ) );
    }

  }

  // Count tracking particles.
  unsigned int nTPforEff = 0;
  unsigned int nTPforAlgEff = 0;
  for (const TP& tp: vTPs) {
    if (tp.useForEff())  nTPforEff++; 
    if (tp.useForAlgEff()) nTPforAlgEff++; 
  }
  profNumTPs_->Fill(1, vTPs.size());
  profNumTPs_->Fill(2, nTPforEff);
  profNumTPs_->Fill(3, nTPforAlgEff);

  // Study efficiency of stubs to pass front-end electronics cuts.

  const vector<Stub>& vAllStubs = inputData.getAllStubs(); // Get all stubs prior to FE cuts to do this.
  for (const Stub s : vAllStubs) {
    unsigned int layerOrTenPlusRing = s.barrel()  ?  s.layerId()  :  10 + s.endcapRing(); 
    // Fraction of all stubs (good and bad) failing tightened front-end electronics cuts.
    hisStubKillFE_->Fill(layerOrTenPlusRing, (! s.frontendPass()));
    // Fraction of stubs rejected by window cut in DegradeBend.h
    // If it is non-zero, then encoding in DegradeBend.h should ideally be changed to make it zero.
    hisStubKillDegradeBend_->Fill(layerOrTenPlusRing, s.stubFailedDegradeWindow());
  }

  // Study efficiency for good stubs of tightened front end-electronics cuts.
  for (const TP& tp : vTPs) {
    if (tp.useForAlgEff()) {// Only bother for stubs that are on TP that we have a chance of reconstructing.
      const vector<const Stub*> stubs = tp.assocStubs();
      for (const Stub* s : stubs) {
        hisStubIneffiVsInvPt_->Fill(1./tp.pt()    , (! s->frontendPass()) );
        hisStubIneffiVsEta_->Fill  (fabs(tp.eta()), (! s->frontendPass()) );
      }
    }
  }

  // Plot stub bend-derived information.
  for (const Stub* stub : vStubs) {
    hisPtStub_->Fill(stub->qOverPt()); 
    hisDelPhiStub_->Fill(stub->dphi()); 
    hisBendStub_->Fill(stub->dphi() / stub->dphiOverBend());
    // Number of bend values merged together by loss of a bit.
    hisNumMergedBend_->Fill(stub->numMergedBend()); 
    // Min. & max allowed q/Pt obtained from stub bend.
    float minQoverPt = max(float(-1./(houghMinPt_)), stub->qOverPt() - stub->qOverPtres());  
    float maxQoverPt = min(float(1./(houghMinPt_)), stub->qOverPt() + stub->qOverPtres());  
    // Frac. of full q/Pt range allowed by stub bend.
    float fracAllowed = (maxQoverPt - minQoverPt)/(2./(houghMinPt_));
    hisBendFilterPower_->Fill(fracAllowed);
    unsigned int layerOrTenPlusRing = stub->barrel()  ?  stub->layerId()  :  10 + stub->endcapRing(); 
    // Also plot bend before & after to degradation.
    if (stub->psModule()) {
      hisBendFEVsLayerOrRingPS_->Fill(layerOrTenPlusRing, stub->bendInFrontend());
      hisBendVsLayerOrRingPS_->Fill(layerOrTenPlusRing, stub->bend());
    } else {
      hisBendFEVsLayerOrRing2S_->Fill(layerOrTenPlusRing, stub->bendInFrontend());
      hisBendVsLayerOrRing2S_->Fill(layerOrTenPlusRing, stub->bend());
    }
  }

  // Look at stub resolution.
  for (const TP& tp: vTPs) {
    if (tp.useForAlgEff()) {
      const vector<const Stub*>& assStubs= tp.assocStubs();
      hisNumStubsPerTP_->Fill( assStubs.size() );

      unsigned int numPSstubs = 0;
      unsigned int num2Sstubs = 0;

      //cout<<"=== TP === : index="<<tp.index()<<" pt="<<tp.pt()<<" q="<<tp.charge()<<" phi="<<tp.phi0()<<" eta="<<tp.eta()<<" z0="<<tp.z0()<<endl;
      for (const Stub* stub: assStubs) {

        if ( stub->psModule() ) ++numPSstubs;
        else ++num2Sstubs;

        //cout<<"    stub : index="<<stub->index()<<" barrel="<<stub->barrel()<<" r="<<stub->r()<<" phi="<<stub->phi()<<" z="<<stub->z()<<" bend="<<stub->bend()<<" assocTP="<<stub->assocTP()->index()<<endl; 
        hisPtResStub_->Fill(stub->qOverPt() - tp.charge()/tp.pt()); 
        hisDelPhiResStub_->Fill(stub->dphi() - tp.dphi(stub->r()));

        if ( stub->moduleTilt() > M_PI / 2 - 0.1  || !stub->barrel() ) {
          hisDelPhiResStub_notTilted_->Fill(stub->dphi() - tp.dphi(stub->r()));
        } else {
          hisDelPhiResStub_tilted_->Fill(stub->dphi() - tp.dphi(stub->r()));
        }
        hisBendResStub_->Fill( (stub->dphi() - tp.dphi(stub->r())) / stub->dphiOverBend() ); 
	// This checks if the TP multiple scattered before producing the stub or hit resolution effects.
        hisPhiStubVsPhiTP_->Fill( reco::deltaPhi(stub->phi(), tp.trkPhiAtStub( stub )) );
	// This checks how wide overlap must be if using phi0 sectors, with no stub bend info used for assignment.
        hisPhiStubVsPhi0TP_->Fill( reco::deltaPhi(stub->phi(), tp.phi0()) );
	// This checks how wide overlap must be if using phi0 sectors, with stub bend info used for assignment
        hisPhi0StubVsPhi0TP_->Fill( reco::deltaPhi(stub->trkPhiAtR(0.).first, tp.phi0()) );
	// This normalizes the previous distribution to the predicted resolution to check if the latter is OK.
        hisPhi0StubVsPhi0TPres_->Fill( reco::deltaPhi(stub->trkPhiAtR(0.).first, tp.phi0()) / stub->trkPhiAtRres(0.));
	// This checks how wide overlap must be if using phi65 sectors, with no stub bend info used for assignment.
        hisPhiStubVsPhi65TP_->Fill( reco::deltaPhi(stub->phi(), tp.trkPhiAtR(65.)) );
	// This checks how wide overlap must be if using phi65 sectors, with stub bend info used for assignment, optionally reducing discrepancy by uncertainty expected from 2S module strip length.
	pair<float, float> phiAndErr = stub->trkPhiAtR(65.);
	double dPhi = reco::deltaPhi( phiAndErr.first, tp.trkPhiAtR(65.));
        hisPhi65StubVsPhi65TP_->Fill( dPhi );
	// This normalizes the previous distribution to the predicted resolution to check if the latter is OK.
        hisPhi65StubVsPhi65TPres_->Fill( dPhi / stub->trkPhiAtRres(65.));

        // Plot B variable
        hisStubB_->Fill( stub->dphiOverBendCorrection() );
        hisStubBApproxDiff_tilted_->Fill( stub->dphiOverBendCorrection() - stub->dphiOverBendCorrectionApprox() );

      }

      hisNumPSStubsPerTP_->Fill( numPSstubs );
      hisNum2SStubsPerTP_->Fill( num2Sstubs );

      if ( fabs( tp.eta() ) < 0.5 ) {
        double nLayersOnTP = Utility::countLayers(settings_, assStubs, true, false);
        double nPSLayersOnTP = Utility::countLayers(settings_, assStubs, true, true);
        hisNumLayersPerTP_->Fill( nLayersOnTP );
        hisNumPSLayersPerTP_->Fill( nPSLayersOnTP );
        hisNum2SLayersPerTP_->Fill( nLayersOnTP - nPSLayersOnTP );      

        if ( fabs( tp.pdgId() ) == 13 ) {
          hisNumLayersPerTP_muons_->Fill( nLayersOnTP );
          hisNumPSLayersPerTP_muons_->Fill( nPSLayersOnTP );
          hisNum2SLayersPerTP_muons_->Fill( nLayersOnTP - nPSLayersOnTP );
        }
        else if ( fabs( tp.pdgId() ) == 11 ) {
          hisNumLayersPerTP_electrons_->Fill( nLayersOnTP );
          hisNumPSLayersPerTP_electrons_->Fill( nPSLayersOnTP );
          hisNum2SLayersPerTP_electrons_->Fill( nLayersOnTP - nPSLayersOnTP );
        }
        else if ( fabs( tp.pdgId() ) == 211 ) {
          hisNumLayersPerTP_pions_->Fill( nLayersOnTP );
          hisNumPSLayersPerTP_pions_->Fill( nPSLayersOnTP );
          hisNum2SLayersPerTP_pions_->Fill( nLayersOnTP - nPSLayersOnTP );
        }

        if ( tp.pt() > 3 && tp.pt() <= 8 ) {
          hisNumLayersPerTP_lowPt_->Fill( nLayersOnTP );
          hisNumPSLayersPerTP_lowPt_->Fill( nPSLayersOnTP );
          hisNum2SLayersPerTP_lowPt_->Fill( nLayersOnTP - nPSLayersOnTP );
        }
        else if ( tp.pt() > 8 && tp.pt() <= 20 ) {
          hisNumLayersPerTP_mediumPt_->Fill( nLayersOnTP );
          hisNumPSLayersPerTP_mediumPt_->Fill( nPSLayersOnTP );
          hisNum2SLayersPerTP_mediumPt_->Fill( nLayersOnTP - nPSLayersOnTP );
        }
        else if ( tp.pt() > 20 ) {
          hisNumLayersPerTP_highPt_->Fill( nLayersOnTP );
          hisNumPSLayersPerTP_highPt_->Fill( nPSLayersOnTP );
          hisNum2SLayersPerTP_highPt_->Fill( nLayersOnTP - nPSLayersOnTP );
        }
      }


    }
  }

  for (const Stub* stub : vStubs) {
    // Note ratio of sensor pitch to separation (needed to understand how many bits this can be packed into).
    hisPitchOverSep_->Fill(stub->pitchOverSep());
    // Also note this same quantity times 1.0 in the barrel or z/r in the endcap. This product is known as "rho".
    float rho = stub->pitchOverSep();
    if ( ! stub->barrel() ) rho *= fabs(stub->z())/stub->r();
    hisRhoParameter_->Fill(rho);
 
    // Check how strip number correlates with phi coordinate relative to module centre. 
    // (Useful for "alpha" correction for non-radial strips in endcap 2S modules).
    float fracPosInModule = (float(2 * stub->iphi()) - float(stub->nstrip())) / float(stub->nstrip());
    float phiFromStrip = 0.5 * stub->width() * fracPosInModule / stub->r();
    if (stub->z() < 0 && (not stub->barrel())) phiFromStrip *= -1;
    if (stub->outerModuleAtSmallerR())         phiFromStrip *= -1; // Module flipped.
    float phiFromStub = reco::deltaPhi(stub->phi(), 0.5*(stub->minPhi() + stub->maxPhi()));
    hisAlphaCheck_->Fill(phiFromStub, phiFromStrip);
 }

  // Check fraction of stubs sharing a common cluster.
  // Loop over both clusters in each stub, so looking for common clusters in seed (0) or correlation (1) sensor of module.
  typedef pair< unsigned int, pair<float, float> > ClusterLocation;
  for (unsigned int iClus = 0; iClus <= 1; iClus++) {
    map<ClusterLocation, unsigned int> commonClusterMap; 
    for (const Stub* stub : vStubs) {
      // Encode detector ID & strip (or pixel) numbers in both dimensions.
      const ClusterLocation loc( stub->idDet(), pair<float, float>(stub->localU_cluster()[iClus], stub->localV_cluster()[iClus]) );
      if (commonClusterMap.find(loc) == commonClusterMap.end()) {
	commonClusterMap[loc] = 1;
      } else {
	commonClusterMap[loc]++;
      }
    }
    unsigned int nShare = 0;
    for (map<ClusterLocation, unsigned int>::const_iterator it = commonClusterMap.begin(); it != commonClusterMap.end(); it++) {
      if (it->second != 1) nShare += it->second; // 2 or more stubs share a cluster at this detid*strip.
    }
    if (iClus == 0) {
      hisFracStubsSharingClus0_->Fill(float(nShare)/float(vStubs.size()));
    } else {
      hisFracStubsSharingClus1_->Fill(float(nShare)/float(vStubs.size()));
    }
  }

  // Determine r (z) range of each barrel layer (endcap wheel).

  for (const Stub* stub : vStubs) {
    unsigned int layer = stub->layerId();
    if (stub->barrel()) {
      // Get range in r of each barrel layer.
      float r = stub->r();
      if (mapBarrelLayerMinR_.find(layer) == mapBarrelLayerMinR_.end()) {
        mapBarrelLayerMinR_[layer] = r;
        mapBarrelLayerMaxR_[layer] = r;
      } else {
        if (mapBarrelLayerMinR_[layer] > r) mapBarrelLayerMinR_[layer] = r;
        if (mapBarrelLayerMaxR_[layer] < r) mapBarrelLayerMaxR_[layer] = r;
      }
    } else {
      layer = layer%10;
      // Range in |z| of each endcap wheel.
      float z = fabs(stub->z());
      if (mapEndcapWheelMinZ_.find(layer) == mapEndcapWheelMinZ_.end()) {
        mapEndcapWheelMinZ_[layer] = z;
        mapEndcapWheelMaxZ_[layer] = z;
      } else {
        if (mapEndcapWheelMinZ_[layer] > z) mapEndcapWheelMinZ_[layer] = z;
        if (mapEndcapWheelMaxZ_[layer] < z) mapEndcapWheelMaxZ_[layer] = z;
      }
    }
  }

  // Determine Range in (r,|z|) of each module type.

  for (const Stub* stub : vStubs) {
    float r = stub->r();
    float z = fabs(stub->z());
    unsigned int modType = stub->digitalStub().moduleType();
    // Do something ugly, as modules in 1-2nd & 3-4th endcap wheels are different to those in wheel 5 ...
    // And boundary between flat & tilted modules in barrel layers 1-3 varies in z.
    if (stub->barrel() && stub->layerId() == 1) { // barrel layer 1
      if (mapExtraAModuleTypeMinR_.find(modType) == mapExtraAModuleTypeMinR_.end()) {
	mapExtraAModuleTypeMinR_[modType] = r;
	mapExtraAModuleTypeMaxR_[modType] = r;
	mapExtraAModuleTypeMinZ_[modType] = z;
	mapExtraAModuleTypeMaxZ_[modType] = z;
      } else {
	if (mapExtraAModuleTypeMinR_[modType] > r) mapExtraAModuleTypeMinR_[modType] = r;
	if (mapExtraAModuleTypeMaxR_[modType] < r) mapExtraAModuleTypeMaxR_[modType] = r;
	if (mapExtraAModuleTypeMinZ_[modType] > z) mapExtraAModuleTypeMinZ_[modType] = z;
	if (mapExtraAModuleTypeMaxZ_[modType] < z) mapExtraAModuleTypeMaxZ_[modType] = z;
      }
    } else if (stub->barrel() && stub->layerId() == 2) { // barrel layer 2
      if (mapExtraBModuleTypeMinR_.find(modType) == mapExtraBModuleTypeMinR_.end()) {
	mapExtraBModuleTypeMinR_[modType] = r;
	mapExtraBModuleTypeMaxR_[modType] = r;
	mapExtraBModuleTypeMinZ_[modType] = z;
	mapExtraBModuleTypeMaxZ_[modType] = z;
      } else {
	if (mapExtraBModuleTypeMinR_[modType] > r) mapExtraBModuleTypeMinR_[modType] = r;
	if (mapExtraBModuleTypeMaxR_[modType] < r) mapExtraBModuleTypeMaxR_[modType] = r;
	if (mapExtraBModuleTypeMinZ_[modType] > z) mapExtraBModuleTypeMinZ_[modType] = z;
	if (mapExtraBModuleTypeMaxZ_[modType] < z) mapExtraBModuleTypeMaxZ_[modType] = z;
      }
    } else if (! stub->barrel() && (stub->layerId()%10 == 1 || stub->layerId()%10 == 2)) { // endcap wheel 1-2
      if (mapExtraCModuleTypeMinR_.find(modType) == mapExtraCModuleTypeMinR_.end()) {
	mapExtraCModuleTypeMinR_[modType] = r;
	mapExtraCModuleTypeMaxR_[modType] = r;
	mapExtraCModuleTypeMinZ_[modType] = z;
	mapExtraCModuleTypeMaxZ_[modType] = z;
      } else {
	if (mapExtraCModuleTypeMinR_[modType] > r) mapExtraCModuleTypeMinR_[modType] = r;
	if (mapExtraCModuleTypeMaxR_[modType] < r) mapExtraCModuleTypeMaxR_[modType] = r;
	if (mapExtraCModuleTypeMinZ_[modType] > z) mapExtraCModuleTypeMinZ_[modType] = z;
	if (mapExtraCModuleTypeMaxZ_[modType] < z) mapExtraCModuleTypeMaxZ_[modType] = z;
      }
    } else if (! stub->barrel() && (stub->layerId()%10 == 3 || stub->layerId()%10 == 4)) { // endcap wheel 3-4
      if (mapExtraDModuleTypeMinR_.find(modType) == mapExtraDModuleTypeMinR_.end()) {
	mapExtraDModuleTypeMinR_[modType] = r;
	mapExtraDModuleTypeMaxR_[modType] = r;
	mapExtraDModuleTypeMinZ_[modType] = z;
	mapExtraDModuleTypeMaxZ_[modType] = z;
      } else {
	if (mapExtraDModuleTypeMinR_[modType] > r) mapExtraDModuleTypeMinR_[modType] = r;
	if (mapExtraDModuleTypeMaxR_[modType] < r) mapExtraDModuleTypeMaxR_[modType] = r;
	if (mapExtraDModuleTypeMinZ_[modType] > z) mapExtraDModuleTypeMinZ_[modType] = z;
	if (mapExtraDModuleTypeMaxZ_[modType] < z) mapExtraDModuleTypeMaxZ_[modType] = z;
      }
    } else { // barrel layer 3-6 or endcap wheel 5.
      if (mapModuleTypeMinR_.find(modType) == mapModuleTypeMinR_.end()) {
	mapModuleTypeMinR_[modType] = r;
	mapModuleTypeMaxR_[modType] = r;
	mapModuleTypeMinZ_[modType] = z;
	mapModuleTypeMaxZ_[modType] = z;
      } else {
	if (mapModuleTypeMinR_[modType] > r) mapModuleTypeMinR_[modType] = r;
	if (mapModuleTypeMaxR_[modType] < r) mapModuleTypeMaxR_[modType] = r;
	if (mapModuleTypeMinZ_[modType] > z) mapModuleTypeMinZ_[modType] = z;
	if (mapModuleTypeMaxZ_[modType] < z) mapModuleTypeMaxZ_[modType] = z;
      }
    }
  }

  //=== Make denominator of tracking efficiency plots

  for (const TP& tp: vTPs) {

    if (tp.useForEff()) { // Check TP is good for efficiency measurement.

      // Check which eta and phi sectors this TP is in.
      int iPhiSec_TP = -1;
      int iEtaReg_TP = -1;
      Sector sectorTmp;
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
	sectorTmp.init(settings_, iPhiSec, 0);
	if (sectorTmp.insidePhiSec(tp)) iPhiSec_TP = iPhiSec;
      }
      for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
	sectorTmp.init(settings_, 0, iEtaReg);
	if (sectorTmp.insideEtaReg(tp)) iEtaReg_TP = iEtaReg;
      }

      // Plot kinematics of all good TP.
      hisTPinvptForEff_->Fill(1./tp.pt());
      hisTPptForEff_->Fill(tp.pt());
      hisTPetaForEff_->Fill(tp.eta());
      hisTPphiForEff_->Fill(tp.phi0());
      // Plot also production point of all good TP.
      hisTPd0ForEff_->Fill(fabs(tp.d0()));
      hisTPz0ForEff_->Fill(fabs(tp.z0()));

      if (tp.useForAlgEff()) { // Check TP is good for algorithmic efficiency measurement.
        hisTPinvptForAlgEff_->Fill(1./tp.pt());
        hisTPptForAlgEff_->Fill(tp.pt());
        hisTPetaForAlgEff_->Fill(tp.eta());
        hisTPphiForAlgEff_->Fill(tp.phi0());
	// Plot also production point of all good TP.
        hisTPd0ForAlgEff_->Fill(fabs(tp.d0()));
        hisTPz0ForAlgEff_->Fill(fabs(tp.z0()));
	// Plot sector nunber.
        hisTPphisecForAlgEff_->Fill(iPhiSec_TP);
        hisTPetasecForAlgEff_->Fill(iEtaReg_TP);

        // Plot 1/pt for TPs inside a jet
        if ( tp.tpInJet() ) {
          hisTPinvptForAlgEff_inJetPtG30_->Fill(1./tp.pt());
        }
        if ( tp.tpInHighPtJet() ) {
          hisTPinvptForAlgEff_inJetPtG100_->Fill(1./tp.pt());           
        }
        if ( tp.tpInVeryHighPtJet() ) {
          hisTPinvptForAlgEff_inJetPtG200_->Fill(1./tp.pt());           
        }
      }
    }
  }
}

//=== Book histograms checking if (eta,phi) sector defis(nition choices are good.

TFileDirectory Histos::bookEtaPhiSectors() {
  TFileDirectory inputDir = fs_->mkdir("CheckSectors");

  // Check if TP lose stubs because not all in same sector.

  hisFracStubsInSec_    = inputDir.make<TH1F>("FracStubsInSec","; Fraction of stubs on TP in best (#eta,#phi) sector;",102,-0.01,1.01);
  hisFracStubsInEtaSec_ = inputDir.make<TH1F>("FracStubsInEtaSec","; Fraction of stubs on TP in best #eta sector;",102,-0.01,1.01);
  hisFracStubsInPhiSec_ = inputDir.make<TH1F>("FracStubsInPhiSec","; Fraction of stubs on TP in best #phi sector;",102,-0.01,1.01);

  // Check if stubs excessively duplicated between overlapping sectors.

  hisNumSecsPerStub_    = inputDir.make<TH1F>("NumSecPerStub","; Number of (#eta,#phi) sectors each stub appears in",20,-0.5,19.5);
  hisNumEtaSecsPerStub_ = inputDir.make<TH1F>("NumEtaSecPerStub","; Number of #eta sectors each stub appears in",20,-0.5,19.5);
  hisNumPhiSecsPerStub_ = inputDir.make<TH1F>("NumPhiSecPerStub","; Number of #phi sectors each stub appears in",20,-0.5,19.5);

  // Count stubs per (eta,phi) sector.
  hisNumStubsPerSec_  = inputDir.make<TH1F>("NumStubsPerSec","; Number of stubs per sector",250,-0.5,249.5);
  // Ditto, summed over all phi. This checks if equal stubs go into each eta region, important for latency.
  unsigned int nEta = numEtaRegions_;
  profNumStubsPerEtaSec_ = inputDir.make<TProfile>("NumStubsPerEtaSec",";#eta sector; Number of stubs per #eta sector",nEta,-0.5,nEta-0.5);

  // Check which tracker layers are present in each eta sector.
  hisLayerIDvsEtaSec_ = inputDir.make<TH2F>("LayerIDvsEtaSec",";#eta sector; layer ID",nEta,-0.5,nEta-0.5,20,0.5,20.5);
  hisLayerIDreducedvsEtaSec_ = inputDir.make<TH2F>("LayerIDreducedvsEtaSec",";#eta sector; reduced layer ID",nEta,-0.5,nEta-0.5,20,0.5,20.5);

  return inputDir;
}

//=== Fill histograms checking if (eta,phi) sector definition choices are good.

void Histos::fillEtaPhiSectors(const InputData& inputData, const matrix<Sector>& mSectors) {

  const vector<const Stub*>& vStubs = inputData.getStubs();
  const vector<TP>&          vTPs   = inputData.getTPs();

  //=== Loop over good tracking particles, looking for the (eta,phi) sector in which each has the most stubs.
  //=== and checking what fraction of its stubs were in this sector.
 
  for (const TP& tp : vTPs) {
    if (tp.useForAlgEff()) {
      unsigned int nStubs = tp.numAssocStubs(); // no. of stubs in this TP.

      // Number of stubs this TP has in best (eta,phi) sector, and also just dividing sectors in phi or just in eta.
      unsigned int nStubsInBestSec = 0; 
      unsigned int nStubsInBestEtaSec = 0; 
      unsigned int nStubsInBestPhiSec = 0; 

      // Loop over (eta, phi) sectors.
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
	for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {

	  const Sector& sector = mSectors(iPhiSec, iEtaReg);

          // Count number of stubs in given tracking particle which are inside this (phi,eta) sector;
          // or inside it if only the eta cuts are applied; or inside it if only the phi cuts are applied.
	  unsigned int nStubsInSec, nStubsInEtaSec, nStubsInPhiSec;
          sector.numStubsInside( tp, nStubsInSec, nStubsInEtaSec, nStubsInPhiSec);

	  // Note best results obtained in any sector.
          nStubsInBestSec    = max( nStubsInBestSec,    nStubsInSec);
          nStubsInBestEtaSec = max( nStubsInBestEtaSec, nStubsInEtaSec);
          nStubsInBestPhiSec = max( nStubsInBestPhiSec, nStubsInPhiSec);
	}
      }

      // Plot fraction of stubs on each TP in its best sector.
      hisFracStubsInSec_->Fill   ( float(nStubsInBestSec)    / float(nStubs) );
      hisFracStubsInEtaSec_->Fill( float(nStubsInBestEtaSec) / float(nStubs) );
      hisFracStubsInPhiSec_->Fill( float(nStubsInBestPhiSec) / float(nStubs) );
    }
  }

  //=== Loop over all stubs, counting how many sectors each one appears in. 
 
  for (const Stub* stub : vStubs) {

    // Number of (eta,phi), phi & eta sectors containing this stub.
    unsigned int nSecs = 0; 
    unsigned int nEtaSecs = 0; 
    unsigned int nPhiSecs = 0; 

    // Loop over (eta, phi) sectors.
    for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
      for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {

	const Sector& sector = mSectors(iPhiSec, iEtaReg);

	// Check if sector contains stub stub, and if so count it.
	// Take care to just use one eta (phi) typical region when counting phi (eta) sectors.
	if ( sector.inside   ( stub ) )                 nSecs++;
	if ( iPhiSec == 0 && sector.insideEta( stub ) ) nEtaSecs++;
	if ( iEtaReg == 0 && sector.insidePhi( stub ) ) nPhiSecs++;

	// Also note which tracker layers are present in each eta sector.
	if (iPhiSec == 0 && sector.insideEta( stub)) {
	  const TP* assocTP = stub->assocTP();
	  if (assocTP != nullptr) {
	    if (assocTP->useForAlgEff()) {
	      unsigned int lay = stub->layerId();
	      if (lay > 20) lay -= 10; // Don't bother distinguishing two endcaps.
	      hisLayerIDvsEtaSec_->Fill(iEtaReg, lay);
	      hisLayerIDreducedvsEtaSec_->Fill(iEtaReg, stub->layerIdReduced()); // Plot also simplified layerID for hardware, which tries to avoid more than 8 ID in any given eta region.
	    }
	  }
	}
      }
    }

    // Plot number of sectors each stub appears in.
    hisNumSecsPerStub_->Fill   ( nSecs );
    hisNumEtaSecsPerStub_->Fill( nEtaSecs );
    hisNumPhiSecsPerStub_->Fill( nPhiSecs );

    if ( ! settings_->allowOver2EtaSecs()) {
      if (nEtaSecs > 2)  throw cms::Exception("Histos ERROR: Stub assigned to more than 2 eta regions. Please redefine eta regions to avoid this!")<<" stub r="<<stub->r()<<" eta="<<stub->eta()<<endl;
    }
  }

  //=== Loop over all sectors, counting the stubs in each one.
  for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
    unsigned int nStubsInEtaSec = 0; // Also counts stubs in eta sector, summed over all phi.
    for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
      const Sector& sector = mSectors(iPhiSec, iEtaReg);

      unsigned int nStubs = 0;
      for (const Stub* stub : vStubs) {
	if ( sector.inside( stub ) )  nStubs++;
      }
      hisNumStubsPerSec_->Fill(nStubs);
      nStubsInEtaSec += nStubs;
    }
    profNumStubsPerEtaSec_->Fill(iEtaReg, nStubsInEtaSec);
  }
}

//=== Book histograms checking filling of r-phi HT array.

TFileDirectory Histos::bookRphiHT() {

  TFileDirectory inputDir = fs_->mkdir("HTrphi");

  // The next block of code is to book a histogram to study unusual HT cell shapes.

  unsigned int shape = settings_->shape();
  float maxAbsQoverPtAxis  = 1. / houghMinPt_; // Max. |q/Pt| covered by  HT array.
  float maxAbsPhiTrkAxis = M_PI / (float)numPhiSectors_; // Half-width of phiTrk axis in HT array.
  float binSizeQoverPtAxis = 2. * maxAbsQoverPtAxis / (float)houghNbinsPt_;
  if ( shape == 2 || shape == 1 || shape == 3 )
    binSizeQoverPtAxis = 2. * maxAbsQoverPtAxis / ( houghNbinsPt_ - 1. );
  float binSizePhiTrkAxis = 2. * maxAbsPhiTrkAxis / (float)houghNbinsPhi_;
  if ( shape == 2 )
    binSizePhiTrkAxis = 2. * maxAbsPhiTrkAxis / ( houghNbinsPhi_ - 1. / 6. );
  else if ( shape == 1 )
    binSizePhiTrkAxis = 2. * maxAbsPhiTrkAxis / ( houghNbinsPhi_ - 1. / 2. );
  hisArrayHT_ = inputDir.make< TH2Poly >( "ArrayHT", "HT Array; m Bins; c Bins",
          -maxAbsQoverPtAxis, maxAbsQoverPtAxis, -maxAbsPhiTrkAxis, maxAbsPhiTrkAxis );
  //hisStubHT_ = inputDir.make< TF1 >( "StubHT", "[0]+[1]*x", -maxAbsQoverPtAxis, maxAbsQoverPtAxis );
  //hisStubHT_->SetMinimum( -maxAbsPhiTrkAxis );
  //hisStubHT_->SetMaximum( maxAbsPhiTrkAxis );
  float xloop, yloop, xtemp;
  Double_t x[7], y[7];
  switch ( shape ) {
    case 0 :
      xloop = - maxAbsQoverPtAxis;
      yloop = - maxAbsPhiTrkAxis;
      for ( unsigned int row = 0; row < houghNbinsPhi_; row++ ) {
        xtemp = xloop;
        for ( unsigned int column = 0; column <  houghNbinsPt_; column++ ) {
          // Go around the square
          x[0] = xtemp;
          y[0] = yloop;
          x[1] = x[0];
          y[1] = y[0] + binSizePhiTrkAxis;
          x[2] = x[1] + binSizeQoverPtAxis;
          y[2] = y[1];
          x[3] = x[2];
          y[3] = y[0];
          x[4] = x[0];
          y[4] = y[0];
          hisArrayHT_->AddBin(5, x, y);
          // Go right
          xtemp += binSizeQoverPtAxis;
        }
        yloop += binSizePhiTrkAxis;
      }
      break;
    case 1 :
      xloop = - maxAbsQoverPtAxis - binSizeQoverPtAxis;
      yloop = - maxAbsPhiTrkAxis;
      for ( unsigned int row = 0; row < houghNbinsPhi_ * 2; row++ ) {
        xtemp = xloop;
        for ( unsigned int column = 0; column <  houghNbinsPt_; column++ ) {
          // Go around the square
          x[0] = xtemp;
          y[0] = yloop;
          x[1] = x[0] + binSizeQoverPtAxis;
          y[1] = y[0] + binSizePhiTrkAxis / 2.;
          x[2] = x[1] + binSizeQoverPtAxis;
          y[2] = y[0];
          x[3] = x[1];
          y[3] = y[0] - binSizePhiTrkAxis / 2.;
          x[4] = x[0];
          y[4] = y[0];
          hisArrayHT_->AddBin(5, x, y);
          // Go right
          xtemp += binSizeQoverPtAxis * 2.;
        }
        xloop += ( row % 2 == 0 ) ? binSizeQoverPtAxis : - binSizeQoverPtAxis;
        yloop += binSizePhiTrkAxis / 2.;
      }
      break;
    case 2 :
      xloop = - maxAbsQoverPtAxis - binSizeQoverPtAxis;
      yloop = - maxAbsPhiTrkAxis;
      for ( unsigned int row = 0; row < houghNbinsPhi_ * 2; row++ ) {
        xtemp = xloop;
        for ( unsigned int column = 0; column <  houghNbinsPt_; column++ ) {
          // Go around the hexagon
          x[0] = xtemp;
          y[0] = yloop;
          x[1] = x[0];
          y[1] = y[0] + binSizePhiTrkAxis / 3.;
          x[2] = x[1] + binSizeQoverPtAxis;
          y[2] = y[1] + binSizePhiTrkAxis / 6.;
          x[3] = x[2] + binSizeQoverPtAxis;
          y[3] = y[1];
          x[4] = x[3];
          y[4] = y[0];
          x[5] = x[2];
          y[5] = y[4] - binSizePhiTrkAxis / 6.;
          x[6] = x[0];
          y[6] = y[0];
          hisArrayHT_->AddBin(7, x, y);
          // Go right
          xtemp += binSizeQoverPtAxis * 2.;
        }
        xloop += ( row % 2 == 0 ) ? binSizeQoverPtAxis : - binSizeQoverPtAxis;
        yloop += binSizePhiTrkAxis / 2.;
      }
      break;
    case 3 :
      xloop = - maxAbsQoverPtAxis - binSizeQoverPtAxis;
      yloop = - maxAbsPhiTrkAxis;
      for ( unsigned int row = 0; row < houghNbinsPhi_ * 2; row++ ) {
        xtemp = xloop;
        for ( unsigned int column = 0; column <  houghNbinsPt_; column++ ) {
          // Go around the square
          x[0] = xtemp;
          y[0] = yloop;
          x[1] = x[0];
          y[1] = y[0] + binSizePhiTrkAxis / 2.;
          x[2] = x[1] + binSizeQoverPtAxis * 2.;
          y[2] = y[1];
          x[3] = x[2];
          y[3] = y[0];
          x[4] = x[0];
          y[4] = y[0];
          hisArrayHT_->AddBin(5, x, y);
          // Go right
          xtemp += binSizeQoverPtAxis * 2.;
        }
        xloop += ( row % 2 == 0 ) ? binSizeQoverPtAxis : - binSizeQoverPtAxis;
        yloop += binSizePhiTrkAxis / 2.;
      }
      break;
  }

  hisIncStubsPerHT_ = inputDir.make<TH1F>("IncStubsPerHT","; Number of filtered stubs per r#phi HT array (inc. duplicates)",100,0.,-1.);
  hisExcStubsPerHT_ = inputDir.make<TH1F>("ExcStubsPerHT","; Number of filtered stubs per r#phi HT array (exc. duplicates)",250,-0.5,249.5);

  hisNumStubsInCellVsEta_ = inputDir.make<TH2F>("NumStubsInCellVsEta","; no. of stubs per HT cell summed over phi sector; #eta region",100,-0.5,499.5, numEtaRegions_, -0.5, numEtaRegions_ - 0.5);

  hisStubsOnRphiTracksPerHT_ = inputDir.make<TH1F>("StubsOnRphiTracksPerHT","; Number of stubs assigned to tracks per r#phi HT array",500,-0.5,499.5);

  hisHTstubsPerTrack_ = inputDir.make<TH1F>("stubsPerTrk","No. stubs per track",25,-0.5,24.5);
  hisHTmBin_ = inputDir.make<TH1F>("mBin","HT m bin", houghNbinsPt_, -0.5, houghNbinsPt_-0.5);
  hisHTcBin_ = inputDir.make<TH1F>("cBin","HT c bin", houghNbinsPhi_, -0.5, houghNbinsPhi_-0.5);

  return inputDir;
}

//=== Fill histograms checking filling of r-phi HT array.

void Histos::fillRphiHT(const matrix<HTrphi>& mHtRphis) {

  //--- Loop over (eta,phi) sectors, counting the number of stubs in the HT array of each.

  if ( plotFirst_ ) {
    const HTrphi& htRphi = mHtRphis(settings_->iPhiPlot(),settings_->iEtaPlot());
    float phiCentreSector =  -M_PI + ( 1. + 2. * settings_->iPhiPlot() ) * M_PI / (float)numPhiSectors_;
    const matrix<HTcell>& htRphiMatrix = htRphi.getAllCells();
    const Stub* stub( nullptr );
    for (unsigned int i = 0; i < htRphiMatrix.size1(); i++)
      for (unsigned int j = 0; j < htRphiMatrix.size2(); j++) {
        std::pair< float, float > cell = htRphi.helix2Dhough(i,j);
        unsigned int numStubs = htRphiMatrix(i,j).numStubs();
        hisArrayHT_->Fill(cell.first, reco::deltaPhi(cell.second, phiCentreSector), numStubs);
        if ( numStubs > 0 )
          stub = htRphiMatrix(i,j).stubs().front();
    }
    if ( stub != nullptr ) {
      //hisStubHT_->SetParameters( reco::deltaPhi( stub->phi(), phiCentreSector ), settings_->invPtToDphi() * ( stub->r() - settings_->chosenRofPhi() ) );
      //hisStubHT_->Draw();
    }
  }
  plotFirst_ = false;

  for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
    for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
      const HTrphi& htRphi = mHtRphis   (iPhiSec, iEtaReg);

      // Here, if a stub appears in multiple cells, it is counted multiple times.
      hisIncStubsPerHT_->Fill( htRphi.numStubsInc() );
      // Here, if a stub appears in multiple cells, it is counted only once.
      hisExcStubsPerHT_->Fill( htRphi.numStubsExc() );
    }
  }

  //--- Count number of stubs in each cell of HT array, summing over all the phi sectors within a given 
  //--- eta region. This determines the buffer size needed to store them in the firmware.

  // Loop over eta regions.
  for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
    // Get dimensions of HT array (assumed same for all phi sectors)
    unsigned int iPhiSecDummy = 0;
    const matrix<HTcell>& rphiHTcellsDummy = mHtRphis(iPhiSecDummy, iEtaReg).getAllCells();
    const unsigned int nbins1 = rphiHTcellsDummy.size1();
    const unsigned int nbins2 = rphiHTcellsDummy.size2();
    // Loop over cells inside HT array
    for (unsigned int m = 0; m < nbins1; m++) {
      for (unsigned int n = 0; n < nbins2; n++) {
  // Loop over phi sectors
  unsigned int nStubsInCellPhiSum = 0;
        for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
          const HTrphi& htRphi = mHtRphis(iPhiSec, iEtaReg);
          const matrix<HTcell>& rphiHTcells = htRphi.getAllCells(); 
          nStubsInCellPhiSum += rphiHTcells(m,n).numStubs();
        }  
  // Plot total number of stubs in this cell, summed over all phi sectors.
        hisNumStubsInCellVsEta_->Fill( nStubsInCellPhiSum, iEtaReg );
      }
    }
  }

  //--- Count number of cells assigned to track candidates by r-phi HT (before any rz filtering 
  //--- or rz HT has been run).
  for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
    for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
      const HTrphi& htRphi = mHtRphis(iPhiSec, iEtaReg);
      hisStubsOnRphiTracksPerHT_->Fill(htRphi.numStubsOnTrackCands2D()); 
      // Also note cell location of HT tracks.
      for (const L1track2D& trk : htRphi.trackCands2D()) {
	hisHTstubsPerTrack_->Fill(trk.getNumStubs());
	hisHTmBin_->Fill(trk.getCellLocationHT().first);
	hisHTcBin_->Fill(trk.getCellLocationHT().second);
      }
    }
  }
}

//=== Book histograms about r-z track filters (or other filters applied after r-phi HT array).

TFileDirectory Histos::bookRZfilters() {

  TFileDirectory inputDir = fs_->mkdir("RZfilters");

  //--- Histograms for Seed Filter
  if (settings_->rzFilterName() == "SeedFilter") {
    // Check number of track seeds that r-z filters must check.
    hisNumSeedCombinations_ = inputDir.make<TH1F>("NumSeedCombinations_","; Number of seed combinations per track cand; no. seeds ; ", 50, -0.5 , 49.5);
    hisNumGoodSeedCombinations_ = inputDir.make<TH1F>("NumGoodSeedCombinations_","; Number of good seed combinations per track cand; ", 30, -0.5 , 29.5);
  }
  return inputDir;
}

//=== Fill histograms about r-z track filters.

void Histos::fillRZfilters(const matrix<Get3Dtracks>& mGet3Dtrks) {

  for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
    for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
      const Get3Dtracks& get3Dtrk = mGet3Dtrks(iPhiSec, iEtaReg);

      //--- Histograms for Seed Filter
      if (settings_->rzFilterName() == "SeedFilter") {
	// Check number of track seeds per sector that r-z "seed" filter checked.
	const vector<unsigned int>  numSeedComb = get3Dtrk.getRZfilter().numSeedCombsPerTrk();
	for (const unsigned int& num : numSeedComb) {
	  hisNumSeedCombinations_->Fill(num) ;
	}
	// Same again, but this time only considering seeds the r-z filters defined as "good".
	const vector<unsigned int>  numGoodSeedComb = get3Dtrk.getRZfilter().numGoodSeedCombsPerTrk();
	for (const unsigned int& num : numGoodSeedComb) {
	  hisNumGoodSeedCombinations_->Fill(num) ;
	}
      }
    }
  }
}

//=== Book histograms studying track candidates found by Hough Transform.

TFileDirectory Histos::bookTrackCands(string tName) {

  // Now book histograms for studying tracking in general.

  // Define lambda function to facilitate adding "tName" to directory & histogram names.
  //auto addn = [tName](string s){ return TString::Format("%s_%s", s.c_str(), tName.c_str()).Data(); };
  auto addn = [tName](string s){ return TString::Format("%s_%s", s.c_str(), tName.c_str()); };

  TFileDirectory inputDir = fs_->mkdir(addn("TrackCands").Data());

  bool TMTT = (tName == "HT" || tName == "RZ");

  // Count tracks in various ways (including/excluding duplicates, excluding fakes ...)
  profNumTrackCands_[tName]  = inputDir.make<TProfile>(addn("NumTrackCands"),"; class; N. of tracks in tracker",7,0.5,7.5);
  profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(7,"TP for eff recoed");
  profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(6,"TP recoed");
  profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(5,"TP recoed x #eta sector dups");
  profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(4,"TP recoed x sector dups");
  profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(2,"TP recoed x track dups");
  profNumTrackCands_[tName]->GetXaxis()->SetBinLabel(1,"reco tracks including fakes");
  profNumTrackCands_[tName]->LabelsOption("d");

  unsigned int nPhi = numPhiSectors_;
  unsigned int nEta = numEtaRegions_;
  float maxAbsQoverPt = 1./houghMinPt_; // Max. |q/Pt| covered by  HT array.
  hisNumTracksVsQoverPt_[tName] = inputDir.make<TH1F>(addn("NumTracksVsQoverPt"),"; Q/Pt; No. of tracks in tracker",100, -maxAbsQoverPt, maxAbsQoverPt);
  hisNumTrksPerNon_[tName]  = inputDir.make<TH1F>(addn("NumTrksPerNon"), "; No. tracks per nonant;",200,-0.5,199.5);
  if (TMTT) {
    profNumTracksVsEta_[tName] = inputDir.make<TProfile>(addn("NumTracksVsEta"),"; #eta region; No. of tracks in tracker", nEta, -0.5, nEta - 0.5);
    hisNumTrksPerSect_[tName] = inputDir.make<TH1F>(addn("NumTrksPerSect"),"; No. tracks per sector;",100,-0.5,99.5);
  }

  // Count stubs per event assigned to tracks (determines HT data output rate)

  profStubsOnTracks_[tName] = inputDir.make<TProfile>(addn("StubsOnTracks"),"; ; No. of stubs on tracks per event",1,0.5,1.5);
  hisStubsOnTracksPerNon_[tName]  = inputDir.make<TH1F>(addn("StubsOnTracksPerNon"),"; No. of stubs on tracks per nonant", 1000,-0.5,999.5); 
  hisUniqueStubsOnTrksPerNon_[tName]  = inputDir.make<TH1F>(addn("UniqueStubsOnTrksPerNon"),"; No. of unique stubs on tracks per nonant", 500,-0.5,499.5); 
  if (TMTT) {
    profStubsOnTracksVsEta_[tName] = inputDir.make<TProfile>(addn("StubsOnTracksVsEta"),"; #eta region; No. of stubs on tracks per event", nEta, -0.5, nEta - 0.5); 
    hisStubsOnTracksPerSect_[tName] = inputDir.make<TH1F>(addn("StubsOnTracksPerSect"),"; No. of stubs on tracks per sector", 500,-0.5,499.5); 
    hisUniqueStubsOnTrksPerSect_[tName] = inputDir.make<TH1F>(addn("UniqueStubsOnTrksPerSect"),"; No. of unique stubs on tracks per sector", 500,-0.5,499.5); 
  }

  hisStubsPerTrack_[tName] = inputDir.make<TH1F>(addn("StubsPerTrack"),";No. of stubs per track;",50,-0.5,49.5);
  hisLayersPerTrack_[tName] = inputDir.make<TH1F>(addn("LayersPerTrack"),";No. of layers with stubs per track;",20,-0.5,19.5);
  hisPSLayersPerTrack_[tName] = inputDir.make<TH1F>(addn("PSLayersPerTrack"),";No. of PS layers with stubs per track;",20,-0.5,19.5);
  hisLayersPerTrueTrack_[tName] = inputDir.make<TH1F>(addn("LayersPerTrueTrack"),";No. of layers with stubs per genuine track;",20,-0.5,19.5);
  hisPSLayersPerTrueTrack_[tName] = inputDir.make<TH1F>(addn("PSLayersPerTrueTrack"),";No. of PS layers with stubs per genuine track;",20,-0.5,19.5);

  if (TMTT) {
    hisNumStubsPerLink_[tName] = inputDir.make<TH1F>(addn("NumStubsPerLink"), "; Mean #stubs per MHT output opto-link;", 50,-0.5,199.5);
    hisNumStubsVsLink_[tName] = inputDir.make<TH2F>(addn("NumStubsVsLink"), "; MHT output opto-link; No. stubs/event", 36, -0.5, 35.5, 20,-0.5,199.5);
    profMeanStubsPerLink_[tName] = inputDir.make<TProfile>(addn("MeanStubsPerLink"), "; Mean #stubs per MHT output opto-link;", 36,-0.5,35.5);
    hisNumTrksPerLink_[tName] = inputDir.make<TH1F>(addn("NumTrksPerLink"), "; Mean #tracks per MHT output opto-link;", 50,-0.5,49.5);
    hisNumTrksVsLink_[tName] = inputDir.make<TH2F>(addn("NumTrksVsLink"), "; MHT output opto-link; No. tracks/event", 72, -0.5, 71.5, 20,-0.5,49.5);
    profMeanTrksPerLink_[tName] = inputDir.make<TProfile>(addn("MeanTrksPerLink"), "; Mean #tracks per MHT output opto-link;", 36,-0.5,35.5);
  }

  if (TMTT) {
    // Checks if tracks have too many stubs to be stored in memory in each cell.
    profExcessStubsPerTrackVsPt_[tName] = inputDir.make<TProfile>(addn("ExcessStubsPerTrackVsPt"),";q/Pt; Prob. of too many stubs per track",16,0.,maxAbsQoverPt);
  }

  hisFracMatchStubsOnTracks_[tName] = inputDir.make<TH1F>(addn("FracMatchStubsOnTracks"),"; Fraction of stubs on tracks matching best TP;",101,-0.005,1.005);

  profFracTrueStubsVsLayer_[tName] = inputDir.make<TProfile>(addn("FracTrueStubsVsLayer"),";Layer ID; fraction of true stubs",30,0.5,30.5);

  // Check how much stub bend differs from predicted one.
  hisDeltaBendTrue_[tName] = inputDir.make<TH1F>(addn("DeltaBendTrue"),"True stubs; stub bend minus true bend / resolution;",100,-2.,2.);
  hisDeltaBendFake_[tName] = inputDir.make<TH1F>(addn("DeltaBendFake"),"Fake stubs; stub bend minus true bend / resolution;",100,-2.,2.);

  if (TMTT) {
    // Study duplication of tracks within HT.
    profDupTracksVsEta_[tName] = inputDir.make<TProfile>(addn("DupTracksVsTPeta"), "; #eta; No. of duplicate tracks per TP in individual HT array;",15,0.0,3.0);
    profDupTracksVsInvPt_[tName] = inputDir.make<TProfile>(addn("DupTracksVsInvPt"), "; 1/Pt; No. of duplicate tracks per TP",16,0.,maxAbsQoverPt);
  }

  // Histos of track params.
  hisQoverPt_[tName]    = inputDir.make<TH1F>(addn("QoverPt"),"; track q/Pt", 100,-0.5,0.5);
  hisPhi0_[tName]       = inputDir.make<TH1F>(addn("Phi0"),   "; track #phi0",70,-3.5,3.5);
  hisEta_[tName]        = inputDir.make<TH1F>(addn("Eta"),    "; track #eta", 70,-3.5,3.5);
  hisZ0_[tName]         = inputDir.make<TH1F>(addn("Z0"),     "; track z0",   100,-25.0,25.0);

  // Histos of track parameter resolution
  hisQoverPtRes_[tName] = inputDir.make<TH1F>(addn("QoverPtRes"),"; track resolution in q/Pt", 100,-0.06,0.06);
  hisPhi0Res_[tName]    = inputDir.make<TH1F>(addn("Phi0Res"),   "; track resolution in #phi0",100,-0.04,0.04);
  hisEtaRes_[tName]     = inputDir.make<TH1F>(addn("EtaRes"),    "; track resolution in #eta", 100,-1.0,1.0);
  hisZ0Res_[tName]      = inputDir.make<TH1F>(addn("Z0Res"),     "; track resolution in z0",   100,-10.0,10.0);

  hisRecoVsTrueQinvPt_[tName] = inputDir.make<TH2F>(addn("RecoVsTrueQinvPt"), "; TP q/p_{T}; Reco q/p_{T} (good #chi^{2})", 60, -0.6, 0.6, 240, -0.6, 0.6 );
  hisRecoVsTruePhi0_[tName]   = inputDir.make<TH2F>(addn("RecoVsTruePhi0"), "; TP #phi_{0}; Reco #phi_{0} (good #chi^{2})", 70, -3.5, 3.5, 280, -3.5, 3.5 );
  hisRecoVsTrueD0_[tName]     = inputDir.make<TH2F>(addn("RecoVsTrueD0"), "; TP d_{0}; Reco d_{0} (good #chi^{2})", 100, -2., 2., 100, -2., 2. );
  hisRecoVsTrueZ0_[tName]     = inputDir.make<TH2F>(addn("RecoVsTrueZ0"), "; TP z_{0}; Reco z_{0} (good #chi^{2})" , 100, -25., 25., 100, -25., 25. );
  hisRecoVsTrueEta_[tName]    = inputDir.make<TH2F>(addn("RecoVsTrueEta"), "; TP #eta; Reco #eta (good #chi^{2})", 70, -3.5, 3.5, 70, -3.5, 3.5 );

  // Histos for tracking efficiency vs. TP kinematics
  hisRecoTPinvptForEff_[tName] = inputDir.make<TH1F>(addn("RecoTPinvptForEff"), "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisRecoTPptForEff_[tName]    = inputDir.make<TH1F>(addn("RecoTPptForEff"), "; Pt of TP (used for effi. measurement);",25,0.0,100.0);
  hisRecoTPetaForEff_[tName]   = inputDir.make<TH1F>(addn("RecoTPetaForEff"),"; #eta of TP (used for effi. measurement);",20,-3.,3.);
  hisRecoTPphiForEff_[tName]   = inputDir.make<TH1F>(addn("RecoTPphiForEff"),"; #phi of TP (used for effi. measurement);",20,-M_PI,M_PI);

  // Histo for efficiency to reconstruct track perfectly (no incorrect hits).
  hisPerfRecoTPinvptForEff_[tName] = inputDir.make<TH1F>(addn("PerfRecoTPinvptForEff"), "; 1/Pt of TP (used for perf. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisPerfRecoTPptForEff_[tName]    = inputDir.make<TH1F>(addn("PerfRecoTPptForEff"), "; Pt of TP (used for perf. effi. measurement);",25,0.0,100.0);
  hisPerfRecoTPetaForEff_[tName]   = inputDir.make<TH1F>(addn("PerfRecoTPetaForEff"),"; #eta of TP (used for perf. effi. measurement);",20,-3.,3.);

  // Histos for  tracking efficiency vs. TP production point
  hisRecoTPd0ForEff_[tName]  = inputDir.make<TH1F>(addn("RecoTPd0ForEff"), "; d0 of TP (used for effi. measurement);",40,0.,4.);
  hisRecoTPz0ForEff_[tName]  = inputDir.make<TH1F>(addn("RecoTPz0ForEff"), "; z0 of TP (used for effi. measurement);",50,0.,25.);

  // Histos for algorithmic tracking efficiency vs. TP kinematics
  hisRecoTPinvptForAlgEff_[tName] = inputDir.make<TH1F>(addn("RecoTPinvptForAlgEff"), "; 1/Pt of TP (used for alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisRecoTPptForAlgEff_[tName]    = inputDir.make<TH1F>(addn("RecoTPptForAlgEff"), "; Pt of TP (used for alg. effi. measurement);",25,0.0,100.0);
  hisRecoTPetaForAlgEff_[tName]   = inputDir.make<TH1F>(addn("RecoTPetaForAlgEff"),"; #eta of TP (used for alg. effi. measurement);",20,-3.,3.);
  hisRecoTPphiForAlgEff_[tName]   = inputDir.make<TH1F>(addn("RecoTPphiForAlgEff"),"; #phi of TP (used for alg. effi. measurement);",20,-M_PI,M_PI);

  // Histos for algorithmic tracking efficiency in jets.
  hisRecoTPinvptForAlgEff_inJetPtG30_[tName] = inputDir.make<TH1F>(addn("RecoTPinvptForAlgEff_inJetPtG30"), "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisRecoTPinvptForAlgEff_inJetPtG100_[tName] = inputDir.make<TH1F>(addn("RecoTPinvptForAlgEff_inJetPtG100"), "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisRecoTPinvptForAlgEff_inJetPtG200_[tName] = inputDir.make<TH1F>(addn("RecoTPinvptForAlgEff_inJetPtG200"), "; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);

  // Histo for efficiency to reconstruct track perfectly (no incorrect hits).
  hisPerfRecoTPinvptForAlgEff_[tName] = inputDir.make<TH1F>(addn("PerfRecoTPinvptForAlgEff"), "; 1/Pt of TP (used for perf. alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
  hisPerfRecoTPptForAlgEff_[tName]    = inputDir.make<TH1F>(addn("PerfRecoTPptForAlgEff"), "; Pt of TP (used for perf. alg. effi. measurement);",25,0.0,100.0);
  hisPerfRecoTPetaForAlgEff_[tName]   = inputDir.make<TH1F>(addn("PerfRecoTPetaForAlgEff"),"; #eta of TP (used for perf. alg. effi. measurement);",20,-3.,3.);

  // Histos for algorithmic tracking efficiency vs. TP production point
  hisRecoTPd0ForAlgEff_[tName]  = inputDir.make<TH1F>(addn("RecoTPd0ForAlgEff"), "; d0 of TP (used for alg. effi. measurement);",40,0.,4.);
  hisRecoTPz0ForAlgEff_[tName]  = inputDir.make<TH1F>(addn("RecoTPz0ForAlgEff"), "; z0 of TP (used for alg. effi. measurement);",50,0.,25.);

  // Histos for algorithmic tracking efficiency vs sector number (to check if looser cuts are needed in certain regions)
  hisRecoTPphisecForAlgEff_[tName]  = inputDir.make<TH1F>(addn("RecoTPphisecForAlgEff"), "; #phi sector of TP (used for alg. effi. measurement);",nPhi,-0.5,nPhi-0.5);
  hisRecoTPetasecForAlgEff_[tName]  = inputDir.make<TH1F>(addn("RecoTPetasecForAlgEff"), "; #eta sector of TP (used for alg. effi. measurement);",nEta,-0.5,nEta-0.5);

  // Histo for efficiency to reconstruct tracks perfectly (no incorrect hits).
  hisPerfRecoTPphisecForAlgEff_[tName]  = inputDir.make<TH1F>(addn("PerfRecoTPphisecForAlgEff"), "; #phi sector of TP (used for perf. alg. effi. measurement);",nPhi,-0.5,nPhi-0.5);
  hisPerfRecoTPetasecForAlgEff_[tName]  = inputDir.make<TH1F>(addn("PerfRecoTPetasecForAlgEff"), "; #eta sector of TP (used for perf. alg. effi. measurement);",nEta,-0.5,nEta-0.5);

  if (TMTT) {
    // For those tracking particles causing the algorithmic efficiency to be below 100%, plot a flag indicating why.
    hisRecoFailureReason_[tName] = inputDir.make<TH1F>(addn("RecoFailureReason"),"; Reason TP (used for alg. effi.) not reconstructed;",1,-0.5,0.5); 
    //hisRecoFailureLayer_[tName] = inputDir.make<TH1F>(addn("RecoFailureLayer"),"; Layer ID of lost stubs on unreconstructed TP;",30,-0.5,29.5);
  }

  //hisWrongSignStubRZ_pBend_[tName] = inputDir.make<TH2F>(addn("WrongSignStubRZ_pBend"),"RZ of stubs with positive bend, but with wrong sign; z (cm); radius (cm); No. stubs in tracker",100,-280,280,100,0,130);
  //hisWrongSignStubRZ_nBend_[tName] = inputDir.make<TH2F>(addn("WrongSignStubRZ_nBend"),"RZ of stubs with negative bend, but with wrong sign; z (cm); radius (cm); No. stubs in tracker",100,-280,280,100,0,130);

  hisNumStubsOnLayer_[tName] = inputDir.make<TH1F>(addn("NumStubsOnLayer"),"; Layer occupancy;",16,1,17); 

  return inputDir;
}

//=== Fill histograms studying track candidates found before track fit is run.

void Histos::fillTrackCands(const InputData& inputData, const vector<L1track3D>& tracks, string tName) {

  bool withRZfilter = (tName == "RZ");

  bool TMTT = (tName == "HT" || tName == "RZ");

  // Now fill histograms for studying tracking in general.

  const vector<TP>&  vTPs = inputData.getTPs();

  // Debug histogram for LR track fitter.
  for (const L1track3D& t : tracks) {
    const std::vector< const Stub* > stubs = t.getStubs();
    std::map< unsigned int, unsigned int > layerMap;
    for ( auto s : stubs )
      layerMap[ s->layerIdReduced() ]++;
    for ( auto l : layerMap )
      hisNumStubsOnLayer_[tName]->Fill( l.second );
  }

  //=== Count track candidates found in the tracker. 

  const unsigned int numPhiNonants = settings_->numPhiNonants();;
  matrix<unsigned int> nTrksPerSec(numPhiSectors_, numEtaRegions_, 0);
  vector<unsigned int> nTrksPerEtaReg(numEtaRegions_, 0);
  vector<unsigned int> nTrksPerNonant(numPhiNonants, 0);
  for (const L1track3D& t : tracks) {
    unsigned int iNonant = floor((t.iPhiSec())*numPhiNonants/(numPhiSectors_)); // phi nonant number
    nTrksPerSec(t.iPhiSec(), t.iEtaReg())++;
    nTrksPerEtaReg[t.iEtaReg()]++;
    nTrksPerNonant[iNonant]++;
  }

  profNumTrackCands_[tName]->Fill(1.0, tracks.size()); // Plot mean number of tracks/event.
  if (TMTT) {
    for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
        hisNumTrksPerSect_[tName]->Fill(nTrksPerSec(iPhiSec, iEtaReg));
      }
      profNumTracksVsEta_[tName]->Fill(iEtaReg, nTrksPerEtaReg[iEtaReg]);
    }
  }
  for (unsigned int iNonant = 0; iNonant < numPhiNonants; iNonant++) {
    hisNumTrksPerNon_[tName]->Fill(nTrksPerNonant[iNonant]);
  } 

  //=== Count stubs per event assigned to track candidates in the Tracker

  unsigned int nStubsOnTracks = 0;
  matrix nStubsOnTracksInSec(numPhiSectors_,numEtaRegions_,0);
  vector<unsigned int> nStubsOnTracksInEtaReg(numEtaRegions_, 0);
  vector<unsigned int> nStubsOnTracksInNonant(numPhiNonants, 0);
  map< pair<unsigned int, unsigned int>, set<const Stub*> > uniqueStubsOnTracksInSect;
  map< unsigned int, set<const Stub*> > uniqueStubsOnTracksInNonant;

  matrix<unsigned int> nStubsOnTrksInSec(numPhiSectors_, numEtaRegions_, 0);
  for (const L1track3D& t : tracks) {
    const vector<const Stub*>& stubs = t.getStubs();
    unsigned int nStubs = stubs.size();
    unsigned int iNonant = floor((t.iPhiSec())*numPhiNonants/(numPhiSectors_)); // phi nonant number
    // Count stubs on all tracks in this sector & nonant.
    nStubsOnTracks+= nStubs;
    nStubsOnTrksInSec(t.iPhiSec(), t.iEtaReg()) += nStubs;
    nStubsOnTracksInEtaReg[t.iEtaReg()] += nStubs;
    nStubsOnTracksInNonant[iNonant] += nStubs;
    // Note unique stubs in sector & nonant.
    uniqueStubsOnTracksInSect[pair<unsigned int, unsigned int>(t.iPhiSec(), t.iEtaReg())].insert(stubs.begin(), stubs.end());
    uniqueStubsOnTracksInNonant[iNonant].insert(stubs.begin(), stubs.end());
  }

  profStubsOnTracks_[tName]->Fill(1.0, nStubsOnTracks);
  if (TMTT) {
    for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
        hisStubsOnTracksPerSect_[tName]->Fill(nStubsOnTrksInSec(iPhiSec, iEtaReg));
        // Plot number of stubs assigned to tracks per sector, never counting each individual stub more than once.
        hisUniqueStubsOnTrksPerSect_[tName]->Fill(uniqueStubsOnTracksInSect[pair(iPhiSec, iEtaReg)].size());
      }
      profStubsOnTracksVsEta_[tName]->Fill(iEtaReg, nStubsOnTracksInEtaReg[iEtaReg]);
    }
  }
  for (unsigned int iNonant = 0; iNonant < numPhiNonants; iNonant++) {
    hisStubsOnTracksPerNon_[tName]->Fill(nStubsOnTracksInNonant[iNonant]);
    // Plot number of stubs assigned to tracks per nonant, never counting each individual stub more than once.
    hisUniqueStubsOnTrksPerNon_[tName]->Fill(uniqueStubsOnTracksInNonant[iNonant].size());
  } 

  // Plot number of tracks & number of stubs per output HT opto-link.

  if (TMTT && not withRZfilter) {
    static bool firstMess = true;
    const unsigned int numPhiSecPerNon = numPhiSectors_/numPhiNonants;
    // Hard-wired bodge
    const unsigned int nLinks = houghNbinsPt_/2; // Hard-wired to number of course HT bins. Check.

    for (unsigned int iPhiNon = 0; iPhiNon < numPhiNonants; iPhiNon++) {
      // Each nonant has a separate set of links.
      vector<unsigned int> stubsToLinkCount(nLinks, 0); // Must use vectors to count links with zero entries.
      vector<unsigned int> trksToLinkCount(nLinks, 0);
      for (const L1track3D& trk : tracks) {
        unsigned int iNonantTrk = floor((trk.iPhiSec())*numPhiNonants/(numPhiSectors_)); // phi nonant number
        if (iPhiNon == iNonantTrk) {
          unsigned int link = trk.optoLinkID();
          if (link < nLinks) {
            stubsToLinkCount[link] += trk.getNumStubs();
            trksToLinkCount[link] += 1;
          } else if (firstMess) {
	    firstMess = false;
	    cout<<endl<<"===== HISTOS MESS UP: Increase size of nLinks ===== "<<link<<endl<<endl;
          }
        }
      }

      for (unsigned int link = 0; link < nLinks; link++) {
        unsigned int nstbs = stubsToLinkCount[link];
        hisNumStubsPerLink_[tName]->Fill(nstbs);
        hisNumStubsVsLink_[tName]->Fill(link, nstbs);
        profMeanStubsPerLink_[tName]->Fill(link, nstbs);
      }

      for (unsigned int link = 0; link < nLinks; link++) {
        unsigned int ntrks = trksToLinkCount[link];
        hisNumTrksPerLink_[tName]->Fill(ntrks);
        hisNumTrksVsLink_[tName]->Fill(link, ntrks);
        profMeanTrksPerLink_[tName]->Fill(link, ntrks);
      }
    }
  }


  // Plot q/pt spectrum of track candidates, and number of stubs/tracks
  for (const L1track3D& trk : tracks) {
    hisNumTracksVsQoverPt_[tName]->Fill(trk.qOverPt()); // Plot reconstructed q/Pt of track cands.
    hisStubsPerTrack_[tName]->Fill(trk.getNumStubs());  // Stubs per track.
    const TP* tp = trk.getMatchedTP();
    if (TMTT) {
      // For genuine tracks, check how often they have too many stubs to be stored in cell memory. (Perhaps worse for high Pt particles in jets?).
      if (tp != nullptr) {
        if (tp->useForAlgEff()) profExcessStubsPerTrackVsPt_[tName]->Fill(1./tp->pt(), trk.getNumStubs() > 16);
      }
    }
    hisLayersPerTrack_[tName]->Fill(trk.getNumLayers()); // Number of reduced layers with stubs per track.
    hisPSLayersPerTrack_[tName]->Fill( Utility::countLayers(settings_, trk.getStubs(), false, true) ); // Number of reduced PS layers with stubs per track.
    // Also plot just for genuine tracks.
    if (tp != nullptr && tp->useForAlgEff()) {
      hisLayersPerTrueTrack_[tName]->Fill(trk.getNumLayers()); // Number of reduced layers with stubs per track.
      hisPSLayersPerTrueTrack_[tName]->Fill( Utility::countLayers(settings_, trk.getStubs(), false, true) ); // Number of reduced PS layers with stubs per track.
    }
  }  

  // Count fraction of stubs on each track matched to a TP that are from same TP.

  for (const L1track3D& trk : tracks) {
    // Only consider tracks that match a tracking particle used for the alg. efficiency measurement.
    const TP* tp = trk.getMatchedTP();
    if (tp != nullptr) {
      if (tp->useForAlgEff()) {
	hisFracMatchStubsOnTracks_[tName]->Fill( trk.getPurity() );

	const vector<const Stub*> stubs = trk.getStubs();
	for (const Stub* s : stubs) {
	  // Was this stub produced by correct truth particle?
	  const set<const TP*> stubTPs = s->assocTPs();
	  bool trueStub = (stubTPs.find(tp) != stubTPs.end());

	  // Fraction of wrong stubs vs. tracker layer.
	  profFracTrueStubsVsLayer_[tName]->Fill(s->layerId(), trueStub);

	  // Check how much stub bend differs from predicted one, relative to nominal bend resolution.
	  float diffBend = (s->qOverPt() - trk.qOverPt()) / s->qOverPtOverBend();
	  if (trueStub) {
	    hisDeltaBendTrue_[tName]->Fill(diffBend/s->bendRes());
	  } else {
	    hisDeltaBendFake_[tName]->Fill(diffBend/s->bendRes());
	  }

	  // Debug printout to understand for matched tracks, how far stubs lie from true particle trajectory
	  // Only prints for tracks with huge number of stubs, to also understand why these tracks exist.
	  //if (trk.getNumStubs() > 20) { 
	  /*
	    if (trk.pt() > 20) { 
	    cout<<"--- Checking how far stubs on matched tracks lie from true particle trajectory. ---"<<endl;
	    cout<<"    Track "<<trk.getPurity()<<" "<<tp->pt()<<" "<<tp->d0()<<endl;
	    float sigPhiR = deltaPhiR/s->sigmaPerp();
	    float sigRorZ = deltaRorZ/s->sigmaPar();
	    string ohoh =  (fabs(sigPhiR) > 5 || fabs(sigRorZ) > 5)  ?  "FAR"  :  "NEAR";
	    if (trueStub) {
	    cout<<"    Real stub "<<ohoh<<" ps="<<s->psModule()<<" bar="<<s->barrel()<<" lay="<<s->layerId()<<" : phi="<<deltaPhiR<<" ("<<sigPhiR<<") rz="<<deltaRorZ<<" ("<<sigRorZ<<")"<<endl;
	    } else {
	    cout<<"    FAKE stub "<<ohoh<<" ps="<<s->psModule()<<" bar="<<s->barrel()<<" lay="<<s->layerId()<<" : phi="<<deltaPhiR<<" ("<<sigPhiR<<") rz="<<deltaRorZ<<" ("<<sigRorZ<<")"<<endl; 
	    }
	    cout<<"        coords="<<s->r()<<" "<<s->phi()<<" "<<s->eta()<<" bend="<<s->bend()<<" iphi="<<s->iphi()<<endl;
	    cout<<"        module="<<s->minR()<<" "<<s->minPhi()<<" "<<s->minZ()<<endl;
	    }
	  */
	}
      }
    }
  }

  // Count total number of tracking particles in the event that were reconstructed,
  // counting also how many of them were reconstructed multiple times (duplicate tracks).

  unsigned int nRecoedTPsForEff = 0; // Total no. of TPs used for the efficiency measurement that were reconstructed as at least one track.
  unsigned int nRecoedTPs = 0; // Total no. of TPs that were reconstructed as at least one track.
  unsigned int nEtaSecsMatchingTPs = 0; // Total no. of eta sectors that all TPs were reconstructed in
  unsigned int nSecsMatchingTPs = 0; // Total no. of eta x phi sectors that all TPs were reconstructed in
  unsigned int nTrksMatchingTPs = 0; // Total no. of tracks that all TPs were reconstructed as

  for (const TP& tp: vTPs) {

    vector<const L1track3D*> matchedTrks;
    for (const L1track3D& trk : tracks) {
      const TP* tpAssoc = trk.getMatchedTP();
      if (tpAssoc != nullptr) {
        if (tpAssoc->index() == tp.index()) matchedTrks.push_back(&trk);
      }
    }
    unsigned int nTrk = matchedTrks.size();

    bool tpRecoed = false;

    if (nTrk > 0) {
      tpRecoed = true;            // This TP was reconstructed at least once in tracker.
      nTrksMatchingTPs += nTrk;   // Increment sum by no. of tracks this TP was reconstructed as

      set<unsigned int> iEtaRegRecoed;
      for (const L1track3D* trk : matchedTrks) iEtaRegRecoed.insert(trk->iEtaReg());
      nEtaSecsMatchingTPs = iEtaRegRecoed.size();

      set< pair<unsigned int, unsigned int> > iSecRecoed;
      for (const L1track3D* trk : matchedTrks) iSecRecoed.insert({trk->iPhiSec(), trk->iEtaReg()});
      nSecsMatchingTPs = iSecRecoed.size();


      if (TMTT) {
	for (const auto& p : iSecRecoed) {
	  unsigned int nTrkInSec = 0;
	  for (const L1track3D* trk : matchedTrks) {
	    if (trk->iPhiSec() == p.first && trk->iEtaReg() == p.second) nTrkInSec++; 
	  }
	  if (nTrkInSec > 0) {
  	    profDupTracksVsEta_[tName]->Fill(fabs(tp.eta()), nTrkInSec - 1); // Study duplication of tracks within an individual HT array.
	    profDupTracksVsInvPt_[tName]->Fill(fabs(tp.qOverPt()), nTrkInSec - 1); // Study duplication of tracks within an individual HT array.
	  }
	} 
      }
    }

    if (tpRecoed) {
      // Increment sum each time a TP is reconstructed at least once inside Tracker
      if (tp.useForEff()) nRecoedTPsForEff++;
      nRecoedTPs++; 
    }
  }

  //--- Plot mean number of tracks/event, counting number due to different kinds of duplicates

  // Plot number of TPs used for the efficiency measurement that are reconstructed. 
  profNumTrackCands_[tName]->Fill(7.0, nRecoedTPsForEff);
  // Plot number of TPs that are reconstructed. 
  profNumTrackCands_[tName]->Fill(6.0, nRecoedTPs);
  // Plot number of TPs that are reconstructed. Count +1 for each eta sector they are reconstructed in.
  profNumTrackCands_[tName]->Fill(5.0, nEtaSecsMatchingTPs);
  // Plot number of TPs that are reconstructed. Count +1 for each (eta,phi) sector they are reconstructed in.
  profNumTrackCands_[tName]->Fill(4.0, nSecsMatchingTPs);
  // Plot number of TP that are reconstructed. Count +1 for each track they are reconstructed as.
  profNumTrackCands_[tName]->Fill(2.0, nTrksMatchingTPs);

  // Histos of track helix params.
  for (const L1track3D& trk : tracks) {
    hisQoverPt_[tName]->Fill(trk.qOverPt());
    hisPhi0_[tName]->Fill(trk.phi0());
    hisEta_[tName]->Fill(trk.eta());
    hisZ0_[tName]->Fill(trk.z0());
  }

  // Histos of track parameter resolution

  for (const TP& tp: vTPs) {

    if ((resPlotOpt_ && tp.useForAlgEff()) || (not resPlotOpt_)) { // Check TP is good for efficiency measurement (& also comes from signal event if requested)

      // For each tracking particle, find the corresponding reconstructed track(s).
      for (const L1track3D& trk : tracks) {
	const TP* tpAssoc = trk.getMatchedTP();
	if (tpAssoc != nullptr) {
	  if (tpAssoc->index() == tp.index()) {
	    hisQoverPtRes_[tName]->Fill(trk.qOverPt() - tp.qOverPt());
	    hisPhi0Res_[tName]->Fill(reco::deltaPhi(trk.phi0(), tp.phi0()));
	    hisEtaRes_[tName]->Fill(trk.eta() - tp.eta());
	    hisZ0Res_[tName]->Fill(trk.z0() - tp.z0());

	    hisRecoVsTrueQinvPt_[tName]->Fill( tp.qOverPt(), trk.qOverPt() );
	    hisRecoVsTruePhi0_[tName]->Fill( tp.phi0(), trk.phi0( ));
	    hisRecoVsTrueD0_[tName]->Fill( tp.d0(), trk.d0() );
	    hisRecoVsTrueZ0_[tName]->Fill( tp.z0(), trk.z0() );
	    hisRecoVsTrueEta_[tName]->Fill( tp.eta(), trk.eta() );
	  }
	}
      }
    }
  }

  //=== Study tracking efficiency by looping over tracking particles.

  for (const TP& tp: vTPs) {

    if (tp.useForEff()) { // Check TP is good for efficiency measurement.

      // Check which eta and phi sectors this TP is in.
      int iPhiSec_TP = -1;
      int iEtaReg_TP = -1;
      Sector sectorTmp;
      for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
	sectorTmp.init(settings_, iPhiSec, 0);
	if (sectorTmp.insidePhiSec(tp)) iPhiSec_TP = iPhiSec;
      }
      for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
	sectorTmp.init(settings_, 0, iEtaReg);
	if (sectorTmp.insideEtaReg(tp)) iEtaReg_TP = iEtaReg;
      }

      // Check if this TP was reconstructed anywhere in the tracker..
      bool tpRecoed = false;
      bool tpRecoedPerfect = false;
      for (const L1track3D& trk : tracks) {
	const TP* tpAssoc = trk.getMatchedTP();
	if (tpAssoc != nullptr) {
	  if (tpAssoc->index() == tp.index()) {
	    tpRecoed = true;
	    if (trk.getPurity() == 1.) tpRecoedPerfect = true; 
	  }
	}
      }

      // If TP was reconstucted by HT, then plot its kinematics.
      if (tpRecoed) {
	hisRecoTPinvptForEff_[tName]->Fill(1./tp.pt());
	hisRecoTPptForEff_[tName]->Fill(tp.pt());
	hisRecoTPetaForEff_[tName]->Fill(tp.eta());
	hisRecoTPphiForEff_[tName]->Fill(tp.phi0());
        // Plot also production point of all good reconstructed TP.
	hisRecoTPd0ForEff_[tName]->Fill(fabs(tp.d0()));
	hisRecoTPz0ForEff_[tName]->Fill(fabs(tp.z0()));
	// Also plot efficiency to perfectly reconstruct the track (no fake hits)
	if (tpRecoedPerfect) {
	  hisPerfRecoTPinvptForEff_[tName]->Fill(1./tp.pt());
	  hisPerfRecoTPptForEff_[tName]->Fill(tp.pt());
	  hisPerfRecoTPetaForEff_[tName]->Fill(tp.eta());
	}
	if (tp.useForAlgEff()) { // Check TP is good for algorithmic efficiency measurement.
	  hisRecoTPinvptForAlgEff_[tName]->Fill(1./tp.pt());
	  hisRecoTPptForAlgEff_[tName]->Fill(tp.pt());
	  hisRecoTPetaForAlgEff_[tName]->Fill(tp.eta());
	  hisRecoTPphiForAlgEff_[tName]->Fill(tp.phi0());
	  // Plot also production point of all good reconstructed TP.
	  hisRecoTPd0ForAlgEff_[tName]->Fill(fabs(tp.d0()));
	  hisRecoTPz0ForAlgEff_[tName]->Fill(fabs(tp.z0()));
	  // Plot sector number to understand if looser cuts are needed in certain eta regions.
	  hisRecoTPphisecForAlgEff_[tName]->Fill(iPhiSec_TP);
	  hisRecoTPetasecForAlgEff_[tName]->Fill(iEtaReg_TP);

	  // Plot efficiency in jets
	  if ( tp.tpInJet() ) {
	    hisRecoTPinvptForAlgEff_inJetPtG30_[tName]->Fill(1./tp.pt());
	  }
	  if ( tp.tpInHighPtJet() ) {
	    hisRecoTPinvptForAlgEff_inJetPtG100_[tName]->Fill(1./tp.pt());           
	  }
	  if ( tp.tpInVeryHighPtJet() ) {
	    hisRecoTPinvptForAlgEff_inJetPtG200_[tName]->Fill(1./tp.pt());           
	  }

	  // Also plot efficiency to perfectly reconstruct the track (no fake hits)
	  if (tpRecoedPerfect) {
	    hisPerfRecoTPinvptForAlgEff_[tName]->Fill(1./tp.pt());
	    hisPerfRecoTPptForAlgEff_[tName]->Fill(tp.pt());
	    hisPerfRecoTPetaForAlgEff_[tName]->Fill(tp.eta());
  	    hisPerfRecoTPphisecForAlgEff_[tName]->Fill(iPhiSec_TP);
	    hisPerfRecoTPetasecForAlgEff_[tName]->Fill(iEtaReg_TP);
	  }
	}
      }
    }
  }

  if (TMTT) {
    // Diagnose reason why not all viable tracking particles were reconstructed.
    const map<const TP*, string> diagnosis = this->diagnoseTracking(inputData.getTPs(), tracks, withRZfilter);
    for (const auto& iter: diagnosis) {
      hisRecoFailureReason_[tName]->Fill(iter.second.c_str(), 1.); // Stores flag indicating failure reason.
    }
  }
}

//=== Understand why not all tracking particles were reconstructed.
//=== Returns list of tracking particles that were not reconstructed and an string indicating why.
//=== Only considers TP used for algorithmic efficiency measurement.

// (If string = "mystery", reason for loss unknown. This may be a result of reconstruction of one 
// track candidate preventing reconstruction of another. e.g. Due to duplicate track removal).

map<const TP*, string> Histos::diagnoseTracking(const vector<TP>& allTPs, const vector<L1track3D>& tracks, 
						bool withRZfilter) const 
{
  map<const TP*, string> diagnosis;

  for (const TP& tp: allTPs) {

    string recoFlag = "unknown";

    if ( tp.useForAlgEff()) { //--- Only consider TP that are reconstructable.

      //--- Check if this TP was reconstructed anywhere in the tracker..
      bool tpRecoed = false;
      for (const L1track3D& trk : tracks) {
	const TP* tpAssoc = trk.getMatchedTP();
	if (tpAssoc != nullptr) {
	  if (tpAssoc->index() == tp.index()) tpRecoed = true;
	}
      }

      if ( tpRecoed) {
       
	recoFlag = "success"; // successfully reconstructed so don't bother studying.

      } else {
        
	//--- Check if TP was still reconstructable after cuts applied to stubs by front-end electronics.
	vector<const Stub*> fePassStubs;
	for (const Stub* s : tp.assocStubs()) {
	  if (s->frontendPass()) fePassStubs.push_back(s);
        }
	bool fePass = ( Utility::countLayers(settings_, fePassStubs) >= genMinStubLayers_ );
	
	if ( ! fePass) {

	  recoFlag = "FE electronics"; // Tracking failed because of front-end electronics cuts.

	} else {

	  //--- Check if assignment to (eta,phi) sectors prevented this TP being reconstruted.
	  bool insideSecPass = false;
	  bool insidePhiSecPass = false;
	  bool insideEtaRegPass = false;
	  unsigned int nLayers = 0;
	  // The next to variables are vectors in case track could be recontructed in more than one sector.
	  vector< vector<const Stub*> > insideSecStubs;
	  vector<Sector> sectorBest;
	  for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
	    for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {

	      Sector sectorTmp;
	      sectorTmp.init(settings_, iPhiSec, iEtaReg);

	      // Get stubs on given tracking particle which are inside this (phi,eta) sector;
	      vector<const Stub*> insideSecStubsTmp;
	      vector<const Stub*> insidePhiSecStubsTmp;
	      vector<const Stub*> insideEtaRegStubsTmp;
	      for (const Stub* s: fePassStubs) {
		if (sectorTmp.inside(s))    insideSecStubsTmp.push_back(s);
		if (sectorTmp.insidePhi(s)) insidePhiSecStubsTmp.push_back(s);
		if (sectorTmp.insideEta(s)) insideEtaRegStubsTmp.push_back(s);
	      }
	      // Check if TP could be reconstructed in this (phi,eta) sector.
	      unsigned int nLayersTmp = Utility::countLayers(settings_, insideSecStubsTmp);
	      if ( nLayersTmp >= genMinStubLayers_ ) {
		insideSecPass = true;
		if (nLayers <= nLayersTmp) {
		  if (nLayers < nLayersTmp) {
		    nLayers = nLayersTmp;
		    insideSecStubs.clear();
		    sectorBest.clear();
		  }
		  insideSecStubs.push_back( insideSecStubsTmp );
		  sectorBest.push_back( sectorTmp );
		}
	      }
	      // Check if TP could be reconstructed in this (phi) sector.
	      unsigned int nLayersPhiTmp = Utility::countLayers(settings_, insidePhiSecStubsTmp);
	      if ( nLayersPhiTmp >= genMinStubLayers_ ) insidePhiSecPass = true;
	      // Check if TP could be reconstructed in this (eta) region.
	      unsigned int nLayersEtaTmp = Utility::countLayers(settings_, insideEtaRegStubsTmp);
	      if ( nLayersEtaTmp >= genMinStubLayers_ ) insideEtaRegPass = true;
	    }
	  }

	  if ( ! insideSecPass) {

	    // Tracking failed because of stub to sector assignment.
	    if ( ! insideEtaRegPass) {
	      recoFlag = "#eta sector"; // failed because of stub assignment to eta region.
	    } else if ( ! insidePhiSecPass) {
	      recoFlag = "#phi sector"; // failed because of stub assignment to phi sector.
	    } else {
	      recoFlag = "sector";      // failed because of stub assignment to (eta,phi) sector.
	    }

	  } else {

	    //--- Check if TP was reconstructed by r-phi Hough transform with its bend filted turned off.

	    // Consider all sectors in which the track might be reconstructed.
	    bool rphiHTunfilteredPass = false;
	    for (unsigned int iSec = 0; iSec < sectorBest.size(); iSec++) {
	      const Sector& secBest = sectorBest[iSec];
	      HTrphi htRphiUnfiltered;
	      htRphiUnfiltered.init(settings_, secBest.iPhiSec(), secBest.iEtaReg(), 
				    secBest.etaMin(), secBest.etaMax(), secBest.phiCentre());
	      htRphiUnfiltered.disableBendFilter(); // Switch off bend filter
	      for (const Stub* s: insideSecStubs[iSec]) {
		// Check which eta subsectors within the sector the stub is compatible with (if subsectors being used).
		const vector<bool> inEtaSubSecs =  secBest.insideEtaSubSecs( s );
		htRphiUnfiltered.store(s, inEtaSubSecs);
	      }
	      htRphiUnfiltered.end();
	      // Check if  r-phi HT with its filters switched off found the track
	      if (htRphiUnfiltered.numTrackCands2D() > 0) rphiHTunfilteredPass = true;
	    }

	    if ( ! rphiHTunfilteredPass ) {

	      recoFlag = "r-#phi HT UNfiltered"; // Tracking failed r-phi HT even with its bend filter turned off.

	    } else {

	      //--- Check if TP was reconstructed by filtered r-phi HT.

	      // Consider all sectors in which the track might be reconstructed.
	      bool rphiHTpass   = false;
	      bool rzFilterPass = false;
	      for (unsigned int iSec = 0; iSec < sectorBest.size(); iSec++) {
  	        const Sector& secBest = sectorBest[iSec];
  	        HTrphi htRphiTmp;
	        htRphiTmp.init(settings_, secBest.iPhiSec(), secBest.iEtaReg(), 
			       secBest.etaMin(), secBest.etaMax(), secBest.phiCentre());
		for (const Stub* s: insideSecStubs[iSec]) {
		  // Check which eta subsectors within the sector the stub is compatible with (if subsectors being used).
		  const vector<bool> inEtaSubSecs =  secBest.insideEtaSubSecs( s );
		  htRphiTmp.store(s, inEtaSubSecs);
		}
		htRphiTmp.end();

		// Check if  r-phi HT found the track
		if (htRphiTmp.numTrackCands2D() > 0) rphiHTpass = true;
		// Check if track r-z filters run after r-phi HT kept track.
		if (rphiHTpass) {
		  // Do so by getting tracks found by r-phi HT and running them through r-z filter.
		  const vector<L1track2D>& trksRphi     = htRphiTmp.trackCands2D();

		  // Initialize utility for making 3D tracks from 2S ones.
                  Get3Dtracks get3DtrkTmp;
		  get3DtrkTmp.init(settings_, secBest.iPhiSec(), secBest.iEtaReg(), 
			            secBest.etaMin(), secBest.etaMax(), secBest.phiCentre());
                 // Convert 2D tracks found by HT to 3D tracks (optionally by running r-z filters & duplicate track removal)
                  get3DtrkTmp.run(trksRphi);
		  if (get3DtrkTmp.trackCands3D(withRZfilter).size() > 0) rzFilterPass = true;
		}
	      }
	    
	      if ( ! rphiHTpass) {

		recoFlag = "r-#phi HT BENDfiltered"; // Tracking failed r-phi HT with its bend filter on.

		//--- Debug printout to understand stubs failing bend filter.
                
		
		  // cout<<"TRACK FAILING BEND FILTER: pt="<<tp.pt()<<" eta="<<tp.eta()<<endl;
    //   bool okIfBendMinus1 = true;
		  // for (unsigned int iSec = 0; iSec < sectorBest.size(); iSec++) {
  		//   if (sectorBest.size() > 1) cout<<" SECTOR "<<iSec<<endl;
  		//   for (const Stub* s: insideSecStubs[iSec]) {
    // 		  float bend = s->bend();
    // 		  float bendRes = s->bendRes();
    // 		  float theory = tp.qOverPt()/s->qOverPtOverBend();
    // 		  cout<<" BEND: measured="<<bend<<" theory="<<theory<<" res="<<bendRes<<endl;
    //       cout << s->r() << " " << s->z() << " " << s->layerId()<<" PS="<<s->psModule()<<" Barrel="<<s->barrel() << endl;

    // 		  if (fabs(bend - theory) > bendRes) {
    //   		  bool cluster0_OK = false;
    //   		  if (s->genuineCluster()[0]) cluster0_OK = (s->assocTPofCluster()[0]->index() == tp.index());
    //   		  bool cluster1_OK = false;
    //   		  if (s->genuineCluster()[1]) cluster1_OK = (s->assocTPofCluster()[1]->index() == tp.index());
    //   		  cout<< "    STUB FAILED: layer="<<s->layerId()<<" PS="<<s->psModule()<<" clusters match="<<cluster0_OK<<" "<<cluster1_OK<<endl;
    //         cout << s->bend() << " " << s->stripPitch() << " " << s->stripPitch() / s->pitchOverSep() << " " << s->dphiOverBend() << " " << s->dphi() << std::endl;
    //         cout << "Min R, Z : " << s->minR() << " " << s->minZ() << std::endl;

    //         if ( fabs( bend * -1.0 - theory ) > bendRes ) {
    //           okIfBendMinus1 = false;
    //         }
    //         else { 
    //           if ( bend > 0 ) hisWrongSignStubRZ_pBend_->Fill( s->z(), s->r() );
    //           else if ( bend < 0 ) hisWrongSignStubRZ_nBend_->Fill( s->z(), s->r() );
    //         }
    // 		  }
  		//   }
		  // }

    //   if ( okIfBendMinus1 ) {
    //     recoFlag = "BEND WRONG SIGN"; // Tracking failed r-phi HT with its bend filter on, but would have passed if bend of stubs had opposite sign.
    //   }


	      } else {
	    
	    
		if ( ! rzFilterPass) {

		  recoFlag = "r-z filter"; // Tracking failed r-z filter.

		} else {

		    recoFlag = "mystery"; // Mystery: logically this tracking particle should have been reconstructed. This may be a result of a duplicate track removal algorithm (or something else where finding one track candidate prevents another being found).
		}
	      }
	    }
	  }
	}
        diagnosis[&tp] = recoFlag;
      }
    }
  }
  return diagnosis;
}

//=== Book histograms studying freak, large events with too many stubs.

TFileDirectory Histos::bookStudyBusyEvents() {

  TFileDirectory inputDir = fs_->mkdir("BusyEvents");

  // Look at (eta, phi) sectors with too many input stubs or too many output (= assigned to tracks) stubs.

  unsigned int nEta = numEtaRegions_;

  hisNumBusySecsInPerEvent_  = inputDir.make<TH1F>("NumBusySecsInPerEvent" ,"; No. sectors with too many input stubs/event" , 20, -0.5, 19.5);
  hisNumBusySecsOutPerEvent_ = inputDir.make<TH1F>("NumBusySecsOutPerEvent","; No. sectors with too many output stubs/event", 20, -0.5, 19.5);
  profFracBusyInVsEtaReg_   = inputDir.make<TProfile>("FracBusyInVsEtaReg" ,"; #eta region; Frac. of sectors with too many input stubs" , nEta, -0.5, nEta-0.5);
  profFracBusyOutVsEtaReg_  = inputDir.make<TProfile>("FracBusyOutVsEtaReg","; #eta region; Frac. of sectors with too many output stubs", nEta, -0.5, nEta-0.5);
  profFracStubsKilledVsEtaReg_ = inputDir.make<TProfile>("FracStubsKilledInVsEtaReg" ,"; #eta region; Frac. of input stubs killed" , nEta, -0.5, nEta-0.5);
  profFracTracksKilledVsEtaReg_ = inputDir.make<TProfile>("FracTracksKilledInVsEtaReg" ,"; #eta region; Frac. of track killed" , nEta, -0.5, nEta-0.5);
  profFracTracksKilledVsInvPt_ = inputDir.make<TProfile>("FracTracksKilledInVsInvPt" ,";1/Pt; Frac. of track killed" , 16, 0.,  1./houghMinPt_);
  profFracTPKilledVsEta_   = inputDir.make<TProfile>("FracTPKilledInVsEta" ,";#eta; Efficiency loss due to busy sectors" , 16, 0.,  settings_->maxStubEta());
  profFracTPKilledVsInvPt_ = inputDir.make<TProfile>("FracTPKilledInVsInvPt" ,";1/Pt; Efficiency loss due to busy sectors" , 16, 0.,  1./houghMinPt_);
  hisNumTPkilledBusySec_ = inputDir.make<TH1F>("NumTPkilledBusySec","; No. of TP killed in each busy sector",30,-0.5,29.5);

  // Compare properties of sectors with/without too many output stubs.

  const vector<string> tnames = {"BusyOutSec", "QuietOutSec"};
  const vector<string> enames = {" in busy output sector", " in quiet output sector"};
  for (unsigned int i = 0; i <= 1; i++) {
    const string tn = tnames[i];
    const string en = enames[i];

    hisNumInputStubs_[tn]     = inputDir.make<TH1F>(("NumInputStubs"+(tn)).c_str(),     ("; No. input stubs"+(en)).c_str(),   250, -0.5, 249.5); 
    hisQoverPtInputStubs_[tn] = inputDir.make<TH1F>(("QoverPtInputStubs"+(tn)).c_str(), ("; q/Pt of input stubs"+(en)).c_str(),   30, 0., 1./houghMinPt_); 
    hisNumOutputStubs_[tn]     = inputDir.make<TH1F>(("NumOutputStubs"+(tn)).c_str(),   ("; No. output stubs"+(en)).c_str(), 1000, -0.5, 999.5); 
    hisNumTracks_[tn]         = inputDir.make<TH1F>(("NumTracks"+(tn)).c_str(),         ("; No. tracks"+(en)).c_str(),        200, -0.5, 199.5);
    hisNumStubsPerTrack_[tn]  = inputDir.make<TH1F>(("NumStubsPerTrack"+(tn)).c_str(),  ("; No. stubs/track"+(en)).c_str(),    50, -0.5, 49.5);
    hisTrackQoverPt_[tn]      = inputDir.make<TH1F>(("TrackQoverPt"+(tn)).c_str(),      ("; Track q/pt"+(en)).c_str(),      30, 0., 1./houghMinPt_);
    hisTrackPurity_[tn]       = inputDir.make<TH1F>(("TrackPurity"+(tn)).c_str(),       ("; Track purity"+(en)).c_str(),      102, -0.01, 1.01);
    hisNumTPphysics_[tn]      = inputDir.make<TH1F>(("NumTPphysics"+(tn)).c_str(),      ("; No. physics TP"+(en)).c_str(),     30, -0.5, 29.5);
    hisNumTPpileup_[tn]       = inputDir.make<TH1F>(("NumTPpileup"+(tn)).c_str(),       ("; No. pileup TP"+(en)).c_str(),      30, -0.5, 29.5);
    hisSumPtTPphysics_[tn]    = inputDir.make<TH1F>(("SumPtTPphysics"+(tn)).c_str(),    ("; Sum Pt physics TP"+(en)).c_str(), 100,  0.0, 100.);
    hisSumPtTPpileup_[tn]     = inputDir.make<TH1F>(("SumPtTPpileup"+(tn)).c_str(),     ("; Sum Pt pileup TP"+(en)).c_str(),  100,  0.0, 100.);
  }

  return inputDir;
}

//=== Fill histograms studying freak, large events with too many stubs at HT.

void Histos::fillStudyBusyEvents(const InputData& inputData, const matrix<Sector>& mSectors, const matrix<HTrphi>& mHtRphis, 
                		 const matrix<Get3Dtracks>& mGet3Dtrks) {

  const bool withRZfilter = false; // Care about events at HT.

  const unsigned int numStubsCut = settings_->busySectorNumStubs();   // No. of stubs per HT array the hardware can output.

  const vector<const Stub*>&  vStubs = inputData.getStubs();
  const vector<TP>&           vTPs   = inputData.getTPs();

  // Create map containing L1 tracks found in whole of tracker together with flag indicating if the
  // track was killed because it was in a busy sector.
  map<const L1track3D*, bool> trksInEntireTracker;

  unsigned int nBusySecIn  = 0;
  unsigned int nBusySecOut = 0;

  for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
    for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
      const Sector& sector = mSectors(iPhiSec, iEtaReg);
      const HTrphi& htRphi          = mHtRphis   (iPhiSec, iEtaReg);
      const Get3Dtracks& get3Dtrk = mGet3Dtrks(iPhiSec, iEtaReg);
      const vector<L1track3D>& tracks = get3Dtrk.trackCands3D(withRZfilter);

      //--- Look for too many stubs input to sector.

      unsigned int nStubsIn = htRphi.nReceivedStubs();
      // Plot fraction of input stubs that would be killed by 36BX period.
      for (unsigned int j = 0; j < nStubsIn; j++) {
	bool kill = (j >= numStubsCut);
	profFracStubsKilledVsEtaReg_->Fill(iEtaReg, kill); 
      }
      bool tooBusyIn = (nStubsIn > numStubsCut);      
      if (tooBusyIn) nBusySecIn++;
      profFracBusyInVsEtaReg_->Fill(iEtaReg, tooBusyIn); // Sector had too many input stubs.
 
      //--- Look for too many stubs assigned to output tracks.

      // Order tracks in increasing order of abs(q/Pt).
      // Use multimap rather than map to do this, as some tracks may have identical q/Pt, and it will store all of them, unlike map.
      multimap<float, const L1track3D*> orderedTrks;
      for (const L1track3D& trk : tracks) {
	orderedTrks.insert( pair<float, const L1track3D*>( fabs(trk.qOverPt()), &trk) );
      }

      // Create map containing L1 tracks found in whole of tracker together with flag indicating if the
      // track was killed because it was in a busy sector.
      map<const L1track3D*, bool> trksInSector;

      // Check how many tracks would be killed by 36BX period, assuming we kill preferentially low Pt ones.
      bool tooBusyOut = false;
      unsigned int nStubsOut          = 0;

      for (const auto& oTrk : orderedTrks) {
	float ptInv = oTrk.first;
	const L1track3D* trk = oTrk.second;
	bool kill = false;
	nStubsOut += trk->getNumStubs();
	if (nStubsOut > numStubsCut) kill = true;

	if (kill) tooBusyOut = true; // Note that some tracks were killed in this sector.

	profFracTracksKilledVsEtaReg_->Fill(iEtaReg, kill); 
	profFracTracksKilledVsInvPt_->Fill(ptInv, kill); 

	// Form a map of all tracks in the entire tracker & also just in this sector, with a flag indicating if they were killed as in a busy sector.
	trksInEntireTracker[trk] = kill;
	trksInSector[trk]        = kill;
      }

      if (tooBusyOut) nBusySecOut++;
      profFracBusyOutVsEtaReg_->Fill(iEtaReg, tooBusyOut); // Sector had too many output stubs.

      //--- Compare properties of sectors with/without too many output stubs.

      const vector<string> tnames = {"BusyOutSec", "QuietOutSec"};

      // Loop over sectors with too many/not too many output stubs.
      for (const string& tn : tnames) {
	if ((tn == "BusyOutSec" && tooBusyOut) || (tn == "QuietOutSec" && (! tooBusyOut))) {

	  hisNumInputStubs_[tn]->Fill(nStubsIn);

	  // Check if q/Pt estimated from stub bend differs in busy & quiet sectors.
          for (const Stub* stub : vStubs) {
    	    if ( sector.inside( stub ) ) hisQoverPtInputStubs_[tn]->Fill(abs(stub->qOverPt()));
          }

	  // Look at reconstructed tracks in this sector.
	  hisNumOutputStubs_[tn]->Fill(nStubsOut);
	  hisNumTracks_[tn]->Fill(tracks.size());
	  for (const L1track3D& trk : tracks) {
	    hisNumStubsPerTrack_[tn]->Fill(trk.getNumStubs());
	    hisTrackQoverPt_[tn]->Fill(trk.qOverPt());
	    hisTrackPurity_[tn]->Fill(trk.getPurity());
	  }

	  // Look at total Pt of truth particles in this sector to understand if it contains a jet.
	  unsigned int num_TP_physics = 0;
	  unsigned int num_TP_pileup  = 0;
	  float sumPt_TP_physics = 0.;
	  float sumPt_TP_pileup  = 0.;
	  for (const TP& tp : vTPs) {
	    bool tpInSector = (fabs(tp.trkPhiAtR(settings_->chosenRofPhi()) - sector.phiCentre()) < sector.sectorHalfWidth() && 
			       tp.trkZAtR(chosenRofZ_) > sector.zAtChosenR_Min() && 
			       tp.trkZAtR(chosenRofZ_) < sector.zAtChosenR_Max());
	    if (tpInSector) {
	      if (tp.physicsCollision()) { // distinguish truth particles from physics collision vs from pileup.
		num_TP_physics++;
		sumPt_TP_physics += tp.pt();
	      } else {
		num_TP_pileup++;
		sumPt_TP_pileup  += tp.pt();
	      }
	    }
	  }
	  hisNumTPphysics_[tn]->Fill(num_TP_physics);
	  hisNumTPpileup_[tn]->Fill(num_TP_pileup);
	  hisSumPtTPphysics_[tn]->Fill(sumPt_TP_physics);
	  hisSumPtTPpileup_[tn]->Fill(sumPt_TP_pileup);
	}
      }

      //--- Count tracking particles lost by killing tracks in individual busy sectors.
      if (tooBusyOut) {
	unsigned int nTPkilled = 0;
	for (const TP& tp: vTPs) {
	  if (tp.useForAlgEff()) { // Check TP is good for algorithmic efficiency measurement.

	    bool tpRecoed = false;
	    bool tpRecoedSurvived = false;
	    for (const auto& trkm : trksInSector) {
	      const L1track3D* trk = trkm.first;
	      bool kill            = trkm.second;
	      if (trk->getMatchedTP() == &tp) {
		tpRecoed = true;                        // Truth particle was reconstructed
		if (! kill) tpRecoedSurvived = true;    // Ditto & reconstructed track wasn't killed by busy sector.
	      }
	    }

	    bool tpKilled = tpRecoed && ( ! tpRecoedSurvived );
	    if (tpKilled) nTPkilled++;
	  }
	}
	hisNumTPkilledBusySec_->Fill(nTPkilled);
      }
    }
  }

  hisNumBusySecsInPerEvent_->Fill(nBusySecIn); // No. of sectors per event with too many input stubs.
  hisNumBusySecsOutPerEvent_->Fill(nBusySecOut); // No. of sectors per event with too many output stubs.

  //--- Check loss in tracking efficiency caused by killing tracks in busy sectors.

  for (const TP& tp: vTPs) {
    if (tp.useForAlgEff()) { // Check TP is good for algorithmic efficiency measurement.

      bool tpRecoed = false;
      bool tpRecoedSurvived = false;
      for (const auto& trkm : trksInEntireTracker) {
	const L1track3D* trk = trkm.first;
	bool kill            = trkm.second;
	if (trk->getMatchedTP() == &tp) {
	  tpRecoed = true;                        // Truth particle was reconstructed
	  if (! kill) tpRecoedSurvived = true;    // Ditto & reconstructed track wasn't killed by busy sector.
	}
      }
      bool tpKilled = tpRecoed && ( ! tpRecoedSurvived );
      profFracTPKilledVsEta_->Fill(fabs(tp.eta()), tpKilled);
      profFracTPKilledVsInvPt_->Fill(fabs(tp.qOverPt()), tpKilled);
    }
  }
}

//=== Book histograms for studying track fitting.

map<string, TFileDirectory> Histos::bookTrackFitting() {
 
  const float maxEta = settings_->maxStubEta();
  const float maxAbsQoverPt = 1./houghMinPt_; // Max. |q/Pt| covered by  HT array.
 
  // Book histograms for 4 and 5 parameter helix fits.

  map<string, TFileDirectory> inputDirMap;

  for (const string& fitName : trackFitters_ ) {

    // Define lambda function to facilitate adding "fitName" histogram names.
    auto addn = [fitName](string s){ return TString::Format("%s_%s", s.c_str(), fitName.c_str()); };

    //std::cout << "Booking histograms for " << fitName << std::endl;
    TFileDirectory inputDir = fs_->mkdir( fitName );
    inputDirMap[fitName] = inputDir;
 
    profNumFitTracks_[fitName] = inputDir.make<TProfile>(addn("NumFitTracks"), "; class; # of fitted tracks", 11, 0.5, 11.5, -0.5, 9.9e6);
    profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(7, "TP for eff fitted");
    profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(6, "TP fitted");
    profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(2, "Fit tracks that are genuine");
    profNumFitTracks_[fitName]->GetXaxis()->SetBinLabel(1, "Fit tracks including fakes");
    profNumFitTracks_[fitName]->LabelsOption("d");

    hisNumFitTrks_[fitName]  = inputDir.make<TH1F>(addn("NumFitTrks"), "; No. fitted tracks in tracker;",200,-0.5,399.5);
    hisNumFitTrksPerNon_[fitName]   = inputDir.make<TH1F>(addn("NumFitTrksPerNon"), "; No. fitted tracks per nonant;",200,-0.5,199.5);
    hisNumFitTrksPerSect_[fitName]  = inputDir.make<TH1F>(addn("NumFitTrksPerSect"), "; No. fitted tracks per sector;",100,-0.5,99.5);

    hisStubsPerFitTrack_[fitName]  = inputDir.make<TH1F>(addn("StubsPerFitTrack"), "; No. of stubs per fitted track",20,-0.5,19.5);
    profStubsOnFitTracks_[fitName] = inputDir.make<TProfile>(addn("StubsOnFitTracks"), "; ; No. of stubs on all fitted tracks per event",1,0.5,1.5);
 
    hisFitQinvPtMatched_[fitName] = inputDir.make<TH1F>(addn("FitQinvPtMatched"),"Fitted q/p_{T} for matched tracks", 120, -0.6, 0.6 );
    hisFitPhi0Matched_[fitName]   = inputDir.make<TH1F>(addn("FitPhi0Matched"), "Fitted #phi_{0} for matched tracks", 70, -3.5, 3.5 );
    hisFitD0Matched_[fitName]     = inputDir.make<TH1F>(addn("FitD0Matched"), "Fitted d_{0} for matched tracks", 100, -2., 2. );
    hisFitZ0Matched_[fitName]     = inputDir.make<TH1F>(addn("FitZ0Matched"), "Fitted z_{0} for matched tracks", 100, -25., 25. );
    hisFitEtaMatched_[fitName]    = inputDir.make<TH1F>(addn("FitEtaMatched"), "Fitted #eta for matched tracks", 70, -3.5, 3.5 );
 
    hisFitQinvPtUnmatched_[fitName] = inputDir.make<TH1F>(addn("FitQinvPtUnmatched"), "Fitted q/p_{T} for unmatched tracks", 120, -0.6, 0.6 );
    hisFitPhi0Unmatched_[fitName]   = inputDir.make<TH1F>(addn("FitPhi0Unmatched"), "Fitted #phi_{0} for unmatched tracks", 70, -3.5, 3.5 );
    hisFitD0Unmatched_[fitName]     = inputDir.make<TH1F>(addn("FitD0Unmatched"), "Fitted d_{0} for unmatched tracks", 100, -2., 2. );
    hisFitZ0Unmatched_[fitName]     = inputDir.make<TH1F>(addn("FitZ0Unmatched"), "Fitted z_{0} for unmatched tracks", 100, -25., 25. );
    hisFitEtaUnmatched_[fitName]    = inputDir.make<TH1F>(addn("FitEtaUnmatched"), "Fitted #eta for unmatched tracks", 70, -3.5, 3.5 );

    const unsigned int nBinsChi2 = 29;
    const float chi2dofBins[nBinsChi2+1] = 
      {0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8,3.2,3.6,4.0,4.5,5.0,6.0,7.0,8.0,9.0,10.0,12.0,14.0,16.0,18.0,20.0,25.0,30.0};
    float chi2Bins[nBinsChi2+1];
    for (unsigned int k = 0; k < nBinsChi2+1; k++) chi2Bins[k] = chi2dofBins[k]*6;

    hisFitChi2Matched_[fitName]    = inputDir.make<TH1F>(addn("FitChi2Matched"), ";#chi^{2};", nBinsChi2, chi2Bins );
    hisFitChi2DofMatched_[fitName] = inputDir.make<TH1F>(addn("FitChi2DofMatched"), ";#chi^{2}/DOF;", nBinsChi2, chi2dofBins );
    hisFitChi2DofRphiMatched_[fitName] = inputDir.make<TH1F>(addn("FitChi2DofRphiMatched"), ";#chi^{2}rphi;", nBinsChi2, chi2Bins );
    hisFitChi2DofRzMatched_[fitName]   = inputDir.make<TH1F>(addn("FitChi2DofRzMatched"), ";#chi^{2}rz/DOF;", nBinsChi2, chi2Bins );
    if (settings_->kalmanAddBeamConstr() && fitName.find("KF5") != string::npos) { // Histograms of chi2 with beam-spot constraint only make sense for 5 param fit.
      hisFitBeamChi2Matched_[fitName]    = inputDir.make<TH1F>(addn("FitBeamChi2Matched"), "; Beam constr #chi^{2};", nBinsChi2, chi2Bins);
      hisFitBeamChi2DofMatched_[fitName] = inputDir.make<TH1F>(addn("FitBeamChi2DofMatched"), ";Beam constr #chi^{2}/DOF;", nBinsChi2, chi2dofBins );
    }
    profFitChi2VsEtaMatched_[fitName]      = inputDir.make<TProfile>(addn("FitChi2VsEtaMatched"), "; #eta; Fit #chi^{2}", 24, 0., maxEta );
    profFitChi2DofVsEtaMatched_[fitName]   = inputDir.make<TProfile>(addn("FitChi2DofVsEtaMatched"), "; #eta; Fit #chi^{2}/dof", 24, 0., maxEta );
    profFitChi2VsInvPtMatched_[fitName]    = inputDir.make<TProfile>(addn("FitChi2VsInvPtMatched"), "; 1/p_{T}; Fit #chi^{2}", 25, 0., maxAbsQoverPt );
    profFitChi2DofVsInvPtMatched_[fitName] = inputDir.make<TProfile>(addn("FitChi2DofVsInvPtMatched"), "; 1/p_{T}; Fit #chi^{2}/dof", 25, 0., maxAbsQoverPt );
    const unsigned int nBinsD0 = 8;
    const float d0Bins[nBinsD0+1]={0.0,0.05,0.10,0.15,0.20,0.3,0.5,1.0,2.0};
    profFitChi2VsTrueD0Matched_[fitName]    = inputDir.make<TProfile>(addn("FitChi2VsTrueD0Matched"), "; true d0 (cm); Fit #chi^{2}", nBinsD0, d0Bins );
    profFitChi2DofVsTrueD0Matched_[fitName] = inputDir.make<TProfile>(addn("FitChi2DofVsTrueD0Matched"), "; true d0 (cm); Fit #chi^{2}/dof", nBinsD0, d0Bins);

    hisFitChi2PerfMatched_[fitName]    = inputDir.make<TH1F>(addn("FitChi2PerfMatched"), ";#chi^{2};", nBinsChi2, chi2Bins );
    hisFitChi2DofPerfMatched_[fitName] = inputDir.make<TH1F>(addn("FitChi2DofPerfMatched"), ";#chi^{2}/DOF;", nBinsChi2, chi2dofBins );
  
    hisFitChi2Unmatched_[fitName]    = inputDir.make<TH1F>(addn("FitChi2Unmatched"), ";#chi^{2};", nBinsChi2, chi2Bins );
    hisFitChi2DofUnmatched_[fitName] = inputDir.make<TH1F>(addn("FitChi2DofUnmatched"), ";#chi^{2}/DOF;", nBinsChi2, chi2dofBins );
    hisFitChi2DofRphiUnmatched_[fitName] = inputDir.make<TH1F>(addn("FitChi2DofRphiUnmatched"), ";#chi^{2}rphi/DOF;", nBinsChi2, chi2Bins );
    hisFitChi2DofRzUnmatched_[fitName]   = inputDir.make<TH1F>(addn("FitChi2DofRzUnmatched"), ";#chi^{2}rz/DOF;", nBinsChi2, chi2Bins );
    if (settings_->kalmanAddBeamConstr() && fitName.find("KF5") != string::npos) { // Histograms of chi2 with beam-spot constraint only make sense for 5 param fit.
      hisFitBeamChi2Unmatched_[fitName]    = inputDir.make<TH1F>(addn("FitBeamChi2Unmatched"), "; Beam constr #Chi^{2};", nBinsChi2, chi2Bins );
      hisFitBeamChi2DofUnmatched_[fitName] = inputDir.make<TH1F>(addn("FitBeamChi2DofUnmatched"), "; Beam constr #Chi^{2}/DOF;", nBinsChi2, chi2dofBins );
    }
    profFitChi2VsEtaUnmatched_[fitName]      = inputDir.make<TProfile>(addn("FitChi2VsEtaUnmatched"), "; #eta; Fit #chi2", 24, 0., maxEta );
    profFitChi2DofVsEtaUnmatched_[fitName]   = inputDir.make<TProfile>(addn("FitChi2DofVsEtaUnmatched"), "; #eta; Fit #chi2/dof", 24, 0., maxEta );
    profFitChi2VsInvPtUnmatched_[fitName]    = inputDir.make<TProfile>(addn("FitChi2VsInvPtUnmatched"), "; 1/p_{T}; Fit #chi2", 25, 0., maxAbsQoverPt );
    profFitChi2DofVsInvPtUnmatched_[fitName] = inputDir.make<TProfile>(addn("FitChi2DofVsInvPtUnmatched"), "; 1/p_{T}; Fit #chi2/dof", 25, 0., maxAbsQoverPt );
 
    profFitChi2VsPurity_[fitName] = inputDir.make<TProfile>(addn("FitChi2VsPurity"), "#Chi^{2} vs stub purity", 102, -0.01, 1.01 );
    profFitChi2DofVsPurity_[fitName] = inputDir.make<TProfile>(addn("FitChi2DofVsPurity"), "#Chi^{2}/DOF vs stub purity", 102, -0.01, 1.01 );

    // Monitoring specific track fit algorithms.
    if (fitName.find("KF") != string::npos) {
      hisKalmanNumUpdateCalls_[fitName] = inputDir.make<TH1F>(addn("KalmanNumUpdateCalls"), "; Calls to KF updator;",100,-0.5,99.5);
      hisKalmanChi2DofSkipLay0Matched_[fitName] = inputDir.make<TH1F>(addn("KalmanChi2DofSkipLay0Matched"), ";#chi^{2} for nSkippedLayers = 0;", nBinsChi2, chi2Bins );
      hisKalmanChi2DofSkipLay1Matched_[fitName] = inputDir.make<TH1F>(addn("KalmanChi2DofSkipLay1Matched"), ";#chi^{2} for nSkippedLayers = 1;", nBinsChi2, chi2Bins );
      hisKalmanChi2DofSkipLay2Matched_[fitName] = inputDir.make<TH1F>(addn("KalmanChi2DofSkipLay2Matched"), ";#chi^{2} for nSkippedLayers = 2;", nBinsChi2, chi2Bins );
      hisKalmanChi2DofSkipLay0Unmatched_[fitName] = inputDir.make<TH1F>(addn("KalmanChi2DofSkipLay0Unmatched"), ";#chi^{2} for nSkippedLayers = 0;", nBinsChi2, chi2Bins );
      hisKalmanChi2DofSkipLay1Unmatched_[fitName] = inputDir.make<TH1F>(addn("KalmanChi2DofSkipLay1Unmatched"), ";#chi^{2} for nSkippedLayers = 1;", nBinsChi2, chi2Bins );
      hisKalmanChi2DofSkipLay2Unmatched_[fitName] = inputDir.make<TH1F>(addn("KalmanChi2DofSkipLay2Unmatched"), ";#chi^{2} for nSkippedLayers = 2;", nBinsChi2, chi2Bins );
    }

    // See how far stubs lie from fitted (or true) trajectory
    hisDeltaPhitruePSbarrel_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhitruePSbarrel"),"PS modules; ##sigma of true stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZtruePSbarrel_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZtruePSbarrel"),"PS modules; ##sigma of true stubs from true traj. in r-z;",100,-5.0,5.0);
    hisDeltaPhitrue2Sbarrel_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhitrue2Sbarrel"),"2S modules; ##sigma of true stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZtrue2Sbarrel_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZtrue2Sbarrel"),"2S modules; ##sigma of true stubs from true traj. in r-z;",100,-5.0,5.0);
    hisDeltaPhitruePSendcap_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhitruePSendcap"),"PS modules; ##sigma of true stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZtruePSendcap_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZtruePSendcap"),"PS modules; ##sigma of true stubs from true traj. in r-z;",100,-5.0,5.0);
    hisDeltaPhitrue2Sendcap_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhitrue2Sendcap"),"2S modules; ##sigma of true stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZtrue2Sendcap_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZtrue2Sendcap"),"2S modules; ##sigma of true stubs from true traj. in r-z;",100,-5.0,5.0);
    hisDeltaPhifakePSbarrel_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhifakePSbarrel"),"PS modules; ##sigma of fake stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZfakePSbarrel_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZfakePSbarrel"),"PS modules; ##sigma of fake stubs from true traj. in r-z;",100,-5.0,5.0);
    hisDeltaPhifake2Sbarrel_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhifake2Sbarrel"),"2S modules; ##sigma of fake stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZfake2Sbarrel_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZfake2Sbarrel"),"2S modules; ##sigma of fake stubs from true traj. in r-z;",100,-5.0,5.0);
    hisDeltaPhifakePSendcap_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhifakePSendcap"),"PS modules; ##sigma of fake stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZfakePSendcap_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZfakePSendcap"),"PS modules; ##sigma of fake stubs from true traj. in r-z;",100,-5.0,5.0);
    hisDeltaPhifake2Sendcap_[fitName]  = inputDir.make<TH1F>(addn("DeltaPhifake2Sendcap"),"2S modules; ##sigma of fake stubs from true traj. in phi;",100,-5.0,5.0);
    hisDeltaRorZfake2Sendcap_[fitName] = inputDir.make<TH1F>(addn("DeltaRorZfake2Sendcap"),"2S modules; ##sigma of fake stubs from true traj. in r-z;",100,-5.0,5.0);
    profRecalcRphiChi2VsEtaTrue1_[fitName]  = inputDir.make<TProfile>(addn("RecalcRphiChi2VsEtaTrue1"), "; #eta; Recalculated r-#phi #chi2 method 1 for matched tracks", 24, 0., maxEta );   
    profRecalcRzChi2VsEtaTrue1_[fitName] = inputDir.make<TProfile>(addn("RecalcRzChi2VsEtaTrue1"), "; #eta; Recalculated r-z #chi2 method 1 for matched tracks", 24, 0., maxEta );   
    profRecalcChi2VsEtaTrue1_[fitName] = inputDir.make<TProfile>(addn("RecalcChi2VsEtaTrue1"), "; #eta; Recalculated #chi2 method 1 for matched tracks", 24, 0., maxEta );   
    profRecalcChi2VsEtaTrue2_[fitName] = inputDir.make<TProfile>(addn("RecalcChi2VsEtaTrue2"), "; #eta; Recalculated #chi2 method 2 for matched tracks", 24, 0., maxEta );   
    profNsigmaPhivsInvPt_[fitName] = inputDir.make<TProfile>(addn("NsigmaPhivsInvPt"),"; 1/Pt; Num #sigma of true stubs from true traj.",16,0.,maxAbsQoverPt); 
    profNsigmaPhivsR_[fitName] = inputDir.make<TProfile>(addn("NsigmaPhivsR"),"; r; Num #sigma of true stubs from true traj.",22,0.,110.); 
    profNsigmaPhivsTanl_[fitName] = inputDir.make<TProfile>(addn("NsigmaPhivsTanl"),"; tan #lambda; Num #sigma of true stubs from true traj.",20,0.,6.); 

    hisFitVsSeedQinvPtMatched_[fitName] = inputDir.make<TH2F>(addn("FitVsSeedQinvPtMatched"), "; Seed q/p_{T} (Genuine Cand); Fitted q/p_{T}", 120, -0.6, 0.6, 120, -0.6, 0.6 );
    hisFitVsSeedPhi0Matched_[fitName]   = inputDir.make<TH2F>(addn("FitVsSeedPhi0Matched"), "; Seed #phi_{0} (Genuine Cand); Fitted #phi_{0}", 70, -3.5, 3.5, 70, -3.5, 3.5 );
    hisFitVsSeedD0Matched_[fitName]     = inputDir.make<TH2F>(addn("FitVsSeedD0Matched"), "; Seed d_{0} (Genuine Cand); Fitted d_{0}", 100, -2., 2., 100, -2., 2. );
    hisFitVsSeedZ0Matched_[fitName]     = inputDir.make<TH2F>(addn("FitVsSeedZ0Matched"), "; Seed z_{0} (Genuine Cand); Fitted z_{0}", 100, -25., 25., 100, -25., 25. );
    hisFitVsSeedEtaMatched_[fitName]    = inputDir.make<TH2F>(addn("FitVsSeedEtaMatched"), "; Seed #eta (Genuine Cand); Fitted #eta", 70, -3.5, 3.5, 70, -3.5, 3.5 ); 
 
    hisFitVsSeedQinvPtUnmatched_[fitName] = inputDir.make<TH2F>(addn("FitVsSeedQinvPtUnmatched"), "; Seed q/p_{T} (Fake Cand); Fitted q/p_{T}", 120, -0.6, 0.6, 120, -0.6, 0.6 );
    hisFitVsSeedPhi0Unmatched_[fitName]   = inputDir.make<TH2F>(addn("FitVsSeedPhi0Unmatched"), "; Seed #phi_{0} (Fake Cand); Fitted #phi_{0}", 70, -3.5, 3.5, 70, -3.5, 3.5 );
    hisFitVsSeedD0Unmatched_[fitName]     = inputDir.make<TH2F>(addn("FitVsSeedD0Unmatched"), "; Seed d_{0} (Fake Cand); Fitted d_{0}", 100, -2., 2., 100, -2., 2. );
    hisFitVsSeedZ0Unmatched_[fitName]     = inputDir.make<TH2F>(addn("FitVsSeedZ0Unmatched"), "; Seed z_{0} (Fake Cand); Fitted z_{0}", 100, -25., 25., 100, -25., 25. );
    hisFitVsSeedEtaUnmatched_[fitName]    = inputDir.make<TH2F>(addn("FitVsSeedEtaUnmatched"), "; Seed #eta (Fake Cand); Fitted #eta", 70, -3.5, 3.5, 70, -3.5, 3.5 );

    hisNumStubsVsPurityMatched_[fitName] = inputDir.make<TH2F>(addn("NumStubsVsPurityMatched"), "; Purity; Number of stubs", 102, -0.01, 1.01, 30, 0.0, 30.0);
    profFitFracTrueStubsVsLayerMatched_[fitName] = inputDir.make<TProfile>(addn("FitFracTrueStubsVsLayerMatched") ,";Layer ID; fraction of true stubs",30,0.5,30.5);
    profFitFracTrueStubsVsEtaMatched_[fitName] = inputDir.make<TProfile>(addn("FitFracTrueStubsVsEtaMatched") ,";#eta; fraction of true stubs",24,0.,3.);

    // Plots of helix param resolution.

    hisFitVsTrueQinvPt_[fitName] = inputDir.make<TH2F>(addn("FitVsTrueQinvPt"), "; TP q/p_{T}; Fitted q/p_{T} (good #chi^{2})", 120, -0.6, 0.6, 120, -0.6, 0.6 );
    hisFitVsTruePhi0_[fitName]   = inputDir.make<TH2F>(addn("FitVsTruePhi0"), "; TP #phi_{0}; Fitted #phi_{0} (good #chi^{2})", 70, -3.5, 3.5, 70, -3.5, 3.5 );
    hisFitVsTrueD0_[fitName]     = inputDir.make<TH2F>(addn("FitVsTrueD0"), "; TP d_{0}; Fitted d_{0} (good #chi^{2})", 100, -2., 2., 100, -2., 2. );
    hisFitVsTrueZ0_[fitName]     = inputDir.make<TH2F>(addn("FitVsTrueZ0"), "; TP z_{0}; Fitted z_{0} (good #chi^{2})" , 100, -25., 25., 100, -25., 25. );
    hisFitVsTrueEta_[fitName]    = inputDir.make<TH2F>(addn("FitVsTrueEta"), "; TP #eta; Fitted #eta (good #chi^{2})", 70, -3.5, 3.5, 70, -3.5, 3.5 );

    hisFitQinvPtRes_[fitName] = inputDir.make<TH1F>(addn("FitQinvPtRes"), "Fitted minus true q/p_{T} (good #chi^{2})", 100, -0.1, 0.1 );
    hisFitPhi0Res_[fitName]   = inputDir.make<TH1F>(addn("FitPhi0Res"), "Fitted minus true #phi_{0} (good #chi^{2})", 100, -0.02, 0.02 );
    hisFitD0Res_[fitName]     = inputDir.make<TH1F>(addn("FitD0Res"), "Fitted minus true d_{0} (good #chi^{2})", 100,  -0.2, 0.2 );
    hisFitZ0Res_[fitName]     = inputDir.make<TH1F>(addn("FitZ0Res"), "Fitted minus true z_{0} (good #chi^{2})", 100, -2., 2. );
    hisFitEtaRes_[fitName]    = inputDir.make<TH1F>(addn("FitEtaRes"), "Fitted minus true #eta (good #chi^{2})", 100, -0.02, 0.02 );

    hisQoverPtResVsTrueEta_[fitName] = inputDir.make<TProfile>(addn("QoverPtResVsTrueEta"), "q/p_{T} resolution; |#eta|; q/p_{T} resolution", 24, 0.0, maxEta);
    hisPhi0ResVsTrueEta_[fitName]    = inputDir.make<TProfile>(addn("PhiResVsTrueEta"), "#phi_{0} resolution; |#eta|; #phi_{0} resolution", 24, 0.0, maxEta);
    hisEtaResVsTrueEta_[fitName]     = inputDir.make<TProfile>(addn("EtaResVsTrueEta"), "#eta resolution; |#eta|; #eta resolution", 24, 0.0, maxEta);
    hisZ0ResVsTrueEta_[fitName]      = inputDir.make<TProfile>(addn("Z0ResVsTrueEta"), "z_{0} resolution; |#eta|; z_{0} resolution", 24, 0.0, maxEta);
    hisD0ResVsTrueEta_[fitName]      = inputDir.make<TProfile>(addn("D0ResVsTrueEta"), "d_{0} resolution; |#eta|; d_{0} resolution", 24, 0.0, maxEta);

    hisQoverPtResVsTrueInvPt_[fitName] = inputDir.make<TProfile>(addn("QoverPtResVsTrueInvPt"), "q/p_{T} resolution; 1/p_{T}; q/p_{T} resolution", 25, 0.0, maxAbsQoverPt);
    hisPhi0ResVsTrueInvPt_[fitName]    = inputDir.make<TProfile>(addn("PhiResVsTrueInvPt"), "#phi_{0} resolution; 1/p_{T}; #phi_{0} resolution", 25, 0.0, maxAbsQoverPt);
    hisEtaResVsTrueInvPt_[fitName]     = inputDir.make<TProfile>(addn("EtaResVsTrueInvPt"), "#eta resolution; 1/p_{T}; #eta resolution", 25, 0.0, maxAbsQoverPt);
    hisZ0ResVsTrueInvPt_[fitName]      = inputDir.make<TProfile>(addn("Z0ResVsTrueInvPt"), "z_{0} resolution; 1/p_{T}; z_{0} resolution", 25, 0.0, maxAbsQoverPt);
    hisD0ResVsTrueInvPt_[fitName]      = inputDir.make<TProfile>(addn("D0ResVsTrueInvPt"), "d_{0} resolution; 1/p_{T}; d_{0} resolution", 25, 0.0, maxAbsQoverPt);

    if (settings_->kalmanAddBeamConstr() && fitName.find("KF5") != string::npos) { // Histograms of resolution with beam-spot constraint only make sense for 5 param fit.
      hisQoverPtResBeamVsTrueEta_[fitName] = inputDir.make<TProfile>(addn("QoverPtResBeamVsTrueEta"), "q/p_{T} resolution with beam constr; |#eta|; q/p_{T} resolution", 24, 0.0, maxEta);
      hisPhi0ResBeamVsTrueEta_[fitName]    = inputDir.make<TProfile>(addn("PhiResBeamVsTrueEta"), "#phi_{0} resolution with beam constr; |#eta|; #phi_{0} resolution", 24, 0.0, maxEta);

      hisQoverPtResBeamVsTrueInvPt_[fitName] = inputDir.make<TProfile>(addn("QoverPtResBeamVsTrueInvPt"), "q/p_{T} resolution with beam constr; 1/p_{T}; q/p_{T} resolution", 25, 0.0, maxAbsQoverPt);
      hisPhi0ResBeamVsTrueInvPt_[fitName]    = inputDir.make<TProfile>(addn("PhiResBeamVsTrueInvPt"), "#phi_{0} resolution with beam constr; 1/p_{T}; #phi_{0} resolution", 25, 0.0, maxAbsQoverPt);
    }

    // Duplicate track histos.
    profDupFitTrksVsEta_[fitName]   = inputDir.make<TProfile>(addn("DupFitTrksVsEta") ,"; #eta; No. of duplicate tracks per TP",12,0.,3.);
    profDupFitTrksVsInvPt_[fitName] = inputDir.make<TProfile>(addn("DupFitTrksVsInvPt") ,"; 1/Pt; No. of duplicate tracks per TP",houghNbinsPt_,0.,maxAbsQoverPt);

    // Histos for tracking efficiency vs. TP kinematics. (Binning must match similar histos in bookTrackCands()).
    hisFitTPinvptForEff_[fitName] = inputDir.make<TH1F>(addn("FitTPinvptForEff") ,"; 1/Pt of TP (used for effi. measurement);",24,0.,1.5*maxAbsQoverPt);
    hisFitTPptForEff_[fitName]    = inputDir.make<TH1F>(addn("FitTPptForEff") ,"; Pt of TP (used for effi. measurement);",25,0.0,100.0);
    hisFitTPetaForEff_[fitName]   = inputDir.make<TH1F>(addn("FitTPetaForEff"),"; #eta of TP (used for effi. measurement);",20,-3.,3.);
    hisFitTPphiForEff_[fitName]   = inputDir.make<TH1F>(addn("FitTPphiForEff"),"; #phi of TP (used for effi. measurement);",20,-M_PI,M_PI);

    // Histo for efficiency to reconstruct track perfectly (no incorrect hits). (Binning must match similar histos in bookTrackCands()).
    hisPerfFitTPinvptForEff_[fitName] = inputDir.make<TH1F>(addn("PerfFitTPinvptForEff") ,"; 1/Pt of TP (used for perf. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
    hisPerfFitTPptForEff_[fitName]    = inputDir.make<TH1F>(addn("PerfFitTPptForEff") ,"; Pt of TP (used for perf. effi. measurement);",25,0.0,100.0);
    hisPerfFitTPetaForEff_[fitName]   = inputDir.make<TH1F>(addn("PerfFitTPetaForEff"),"; #eta of TP (used for perfect effi. measurement);",20,-3.,3.);

    // Histos for tracking efficiency vs. TP production point. (Binning must match similar histos in bookTrackCands()).
    hisFitTPd0ForEff_[fitName]   = inputDir.make<TH1F>(addn("FitTPd0ForEff"),"; d0 of TP (used for effi. measurement);",40, 0.,4.);
    hisFitTPz0ForEff_[fitName]   = inputDir.make<TH1F>(addn("FitTPz0ForEff"),"; z0 of TP (used for effi. measurement);",50,0.,25.);

    // Histos for algorithmic tracking efficiency vs. TP kinematics. (Binning must match similar histos in bookTrackCands()).
    hisFitTPinvptForAlgEff_[fitName] = inputDir.make<TH1F>(addn("FitTPinvptForAlgEff") ,"; 1/Pt of TP (used for alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
    hisFitTPptForAlgEff_[fitName]    = inputDir.make<TH1F>(addn("FitTPptForAlgEff") ,"; Pt of TP (used for alg. effi. measurement);",25,0.0,100.0);
    hisFitTPetaForAlgEff_[fitName]   = inputDir.make<TH1F>(addn("FitTPetaForAlgEff"),"; #eta of TP (used for alg. effi. measurement);",20,-3.,3.);
    hisFitTPphiForAlgEff_[fitName]   = inputDir.make<TH1F>(addn("FitTPphiForAlgEff"),"; #phi of TP (used for alg. effi. measurement);",20,-M_PI,M_PI);

    // Histo for efficiency to reconstruct track perfectly (no incorrect hits). (Binning must match similar histos in bookTrackCands()).
    hisPerfFitTPinvptForAlgEff_[fitName] = inputDir.make<TH1F>(addn("PerfFitTPinvptForAlgEff") ,"; 1/Pt of TP (used for perf. alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
    hisPerfFitTPptForAlgEff_[fitName]    = inputDir.make<TH1F>(addn("PerfFitTPptForAlgEff") ,"; Pt of TP (used for perf. alg. effi. measurement);",25,0.0,100.0);
    hisPerfFitTPetaForAlgEff_[fitName]   = inputDir.make<TH1F>(addn("PerfFitTPetaForAlgEff"),"; #eta of TP (used for perf. alg. effi. measurement);",20,-3.,3.);

    // Ditto for tracks inside jets.
    hisPerfFitTPinvptForAlgEff_inJetPtG30_[fitName] = inputDir.make<TH1F>(addn("PerfFitTPinvptForAlgEff_inJetPtG30") ,"; 1/Pt of TP (used for perf. alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
    hisPerfFitTPinvptForAlgEff_inJetPtG100_[fitName] = inputDir.make<TH1F>(addn("PerfFitTPinvptForAlgEff_inJetPtG100") ,"; 1/Pt of TP (used for perf. alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);
    hisPerfFitTPinvptForAlgEff_inJetPtG200_[fitName] = inputDir.make<TH1F>(addn("PerfFitTPinvptForAlgEff_inJetPtG200") ,"; 1/Pt of TP (used for perf. alg. effi. measurement);",24,0.,1.5*maxAbsQoverPt);

    // Histos for algorithmic tracking efficiency vs. TP production point. (Binning must match similar histos in bookTrackCands()).
    hisFitTPd0ForAlgEff_[fitName]  = inputDir.make<TH1F>(addn("FitTPd0ForAlgEff") ,"; d0 of TP (used for alg. effi. measurement);",40,0.,4.);
    hisFitTPz0ForAlgEff_[fitName]  = inputDir.make<TH1F>(addn("FitTPz0ForAlgEff") ,"; z0 of TP (used for alg. effi. measurement);",50,0.,25.);

    // Histo for algorithmic tracking efficiency vs sector number (to check if looser cuts are needed in certain regions)
    unsigned int nPhi = numPhiSectors_;
    unsigned int nEta = numEtaRegions_;
    hisFitTPphisecForAlgEff_[fitName]  = inputDir.make<TH1F>(addn("FitTPphisecForAlgEff") ,"; #phi sector of TP (used for alg. effi. measurement);",nPhi,-0.5,nPhi-0.5);
    hisFitTPetasecForAlgEff_[fitName]  = inputDir.make<TH1F>(addn("FitTPetasecForAlgEff") ,"; #eta sector of TP (used for alg. effi. measurement);",nEta,-0.5,nEta-0.5);
    hisPerfFitTPphisecForAlgEff_[fitName]  = inputDir.make<TH1F>(addn("PerfFitTPphisecForAlgEff") ,"; #phi sector of TP (used for perf. alg. effi. measurement);",nPhi,-0.5,nPhi-0.5);
    hisPerfFitTPetasecForAlgEff_[fitName]  = inputDir.make<TH1F>(addn("PerfFitTPetasecForAlgEff") ,"; #eta sector of TP (used for perf. alg. effi. measurement);",nEta,-0.5,nEta-0.5);
  }

  return inputDirMap;
}

//=== Fill histograms for studying track fitting.

void Histos::fillTrackFitting( const InputData& inputData, const map<string,vector<L1fittedTrack>>& mFittedTracks) {  

  const vector<TP>&  vTPs = inputData.getTPs();
 
  // Loop over all the fitting algorithms we are trying.
  for (const string& fitName : trackFitters_) {

    const vector<L1fittedTrack>& fittedTracks = mFittedTracks.at(fitName); // Get fitted tracks.

    // Count tracks
    unsigned int nFitTracks = 0;
    unsigned int nFitsMatchingTP = 0;

    const unsigned int numPhiNonants = settings_->numPhiNonants();
    vector<unsigned int> nFitTracksPerNonant(numPhiNonants,0);
    map<pair<unsigned int, unsigned int>, unsigned int> nFitTracksPerSector;

    for (const L1fittedTrack& fitTrk : fittedTracks) {
      nFitTracks++;
      // Get matched truth particle, if any.
      const TP* tp = fitTrk.getMatchedTP();
      if (tp != nullptr) nFitsMatchingTP++;
      // Count fitted tracks per nonant.
      unsigned int iNonant = ( numPhiSectors_ > 0 ) ? floor(fitTrk.iPhiSec()*numPhiNonants/(numPhiSectors_)) : 0; // phi nonant number
      nFitTracksPerNonant[iNonant]++;
      nFitTracksPerSector[pair<unsigned int, unsigned int>(fitTrk.iPhiSec(), fitTrk.iEtaReg())]++;
    }

    profNumFitTracks_[fitName]->Fill(1, nFitTracks);
    profNumFitTracks_[fitName]->Fill(2, nFitsMatchingTP);

    hisNumFitTrks_[fitName]->Fill(nFitTracks);
    for (const unsigned int& num : nFitTracksPerNonant) {
      hisNumFitTrksPerNon_[fitName]->Fill(num);
    }
    for (const auto& p : nFitTracksPerSector) {
      hisNumFitTrksPerSect_[fitName]->Fill(p.second);
    }

    // Count stubs assigned to fitted tracks.
    unsigned int nTotStubs = 0;
    for (const L1fittedTrack& fitTrk : fittedTracks) {
      unsigned int nStubs = fitTrk.getNumStubs();
      hisStubsPerFitTrack_[fitName]->Fill(nStubs);
      nTotStubs += nStubs;
    }
    profStubsOnFitTracks_[fitName]->Fill(1., nTotStubs);

    // Note truth particles that are successfully fitted. And which give rise to duplicate tracks.

    map<const TP*, bool> tpRecoedMap; // Note which truth particles were successfully fitted.
    map<const TP*, bool> tpPerfRecoedMap; // Note which truth particles were successfully fitted with no incorrect hits.
    map<const TP*, unsigned int> tpRecoedDup; // Note that this TP gave rise to duplicate tracks. 
    for (const TP& tp: vTPs) {
      tpRecoedMap[&tp]     = false;
      tpPerfRecoedMap[&tp] = false;
      unsigned int nMatch = 0;
      for (const L1fittedTrack& fitTrk : fittedTracks) {
	const TP*   assocTP =  fitTrk.getMatchedTP(); // Get the TP the fitted track matches to, if any.
	if (assocTP == &tp) {
	  tpRecoedMap[&tp] = true; 
	  if (fitTrk.getPurity() == 1.) tpPerfRecoedMap[&tp] = true;
	  nMatch++;
	}
      }
      tpRecoedDup[&tp] = nMatch;
    }

    // Count truth particles that are successfully fitted.

    unsigned int nFittedTPs = 0;
    unsigned int nFittedTPsForEff = 0;
    for (const TP& tp: vTPs) {
      if (tpRecoedMap[&tp]) { // Was this truth particle successfully fitted?
	nFittedTPs++;
	if (tp.useForEff()) nFittedTPsForEff++;
      }    
    }
    
    profNumFitTracks_[fitName]->Fill(6, nFittedTPs);
    profNumFitTracks_[fitName]->Fill(7, nFittedTPsForEff);

    // Loop over fitted tracks again.

    for (const L1fittedTrack& fitTrk : fittedTracks) {

      // Info for specific track fit algorithms.
      unsigned int nSkippedLayers = 0; 
      unsigned int numUpdateCalls = 0;
      if (fitName.find("KF") != string::npos) {
	fitTrk.getInfoKF(nSkippedLayers, numUpdateCalls);
	hisKalmanNumUpdateCalls_[fitName]->Fill(numUpdateCalls);
      }

      //--- Compare fitted tracks that match truth particles to those that don't.

      // Get original HT track candidate prior to fit for comparison.
      const L1track3D& htTrk = fitTrk.getL1track3D();

      // Get matched truth particle, if any.
      const TP* tp = fitTrk.getMatchedTP();

      if (tp != nullptr) {
	hisFitQinvPtMatched_[fitName]->Fill( fitTrk.qOverPt() );
	hisFitPhi0Matched_[fitName]->Fill( fitTrk.phi0() );
	hisFitD0Matched_[fitName]->Fill( fitTrk.d0() );
	hisFitZ0Matched_[fitName]->Fill( fitTrk.z0() );
	hisFitEtaMatched_[fitName]->Fill( fitTrk.eta() );
 
	hisFitChi2Matched_[fitName]->Fill( fitTrk.chi2() );
	hisFitChi2DofMatched_[fitName]->Fill( fitTrk.chi2dof() );
	hisFitChi2DofRphiMatched_[fitName]->Fill( fitTrk.chi2rphi() / fitTrk.numDOFrphi());
	hisFitChi2DofRzMatched_[fitName]->Fill( fitTrk.chi2rz() / fitTrk.numDOFrz());
	if (settings_->kalmanAddBeamConstr() && fitName.find("KF5") != string::npos) { // Histograms of chi2 with beam-spot constraint only make sense for 5 param fit.
	  hisFitBeamChi2Matched_[fitName]->Fill( fitTrk.chi2_bcon() );
	  hisFitBeamChi2DofMatched_[fitName]->Fill( fitTrk.chi2dof_bcon() );
	}
	profFitChi2VsEtaMatched_[fitName]->Fill( fabs(fitTrk.eta()), fitTrk.chi2() );
	profFitChi2DofVsEtaMatched_[fitName]->Fill( fabs(fitTrk.eta()), fitTrk.chi2dof() );
	profFitChi2VsInvPtMatched_[fitName]->Fill( fabs(fitTrk.qOverPt()), fitTrk.chi2() );
	profFitChi2DofVsInvPtMatched_[fitName]->Fill( fabs(fitTrk.qOverPt()), fitTrk.chi2dof() );
	profFitChi2VsTrueD0Matched_[fitName]->Fill( fabs(tp->d0()), fitTrk.chi2() );
	profFitChi2DofVsTrueD0Matched_[fitName]->Fill( fabs(tp->d0()), fitTrk.chi2dof() );

	// Check chi2/dof for perfectly reconstructed tracks.
	if (fitTrk.getPurity() == 1.) {
	  hisFitChi2PerfMatched_[fitName]->Fill( fitTrk.chi2() );
	  hisFitChi2DofPerfMatched_[fitName]->Fill( fitTrk.chi2dof() );
	}

	if (fitName.find("KF") != string::npos) {
	  // No. of skipped layers on track during Kalman track fit.
	  if (nSkippedLayers == 0) {
	    hisKalmanChi2DofSkipLay0Matched_[fitName]->Fill(fitTrk.chi2dof());
	  } else if (nSkippedLayers == 1) {
	    hisKalmanChi2DofSkipLay1Matched_[fitName]->Fill(fitTrk.chi2dof());
	  } else if (nSkippedLayers >= 2) {
	    hisKalmanChi2DofSkipLay2Matched_[fitName]->Fill(fitTrk.chi2dof());
	  }
	}

	// Compared fitted track helix params with seed track from HT.
	hisFitVsSeedQinvPtMatched_[fitName]->Fill( htTrk.qOverPt(), fitTrk.qOverPt() );
	hisFitVsSeedPhi0Matched_[fitName]->Fill( htTrk.phi0(), fitTrk.phi0() );
	hisFitVsSeedD0Matched_[fitName]->Fill( htTrk.d0(), fitTrk.d0() );
	hisFitVsSeedZ0Matched_[fitName]->Fill( htTrk.z0(), fitTrk.z0() );
	hisFitVsSeedEtaMatched_[fitName]->Fill( htTrk.eta(), fitTrk.eta() );

	// Study incorrect hits on matched tracks.
	hisNumStubsVsPurityMatched_[fitName]->Fill( fitTrk.getNumStubs(), fitTrk.getPurity() );

	const vector<const Stub*> stubs = fitTrk.getStubs();
	for (const Stub* s : stubs) {
	  // Was this stub produced by correct truth particle?
	  const set<const TP*> stubTPs = s->assocTPs();
	  bool trueStub = (stubTPs.find(tp) != stubTPs.end());
	  profFitFracTrueStubsVsLayerMatched_[fitName]->Fill(s->layerId(), trueStub);
	  profFitFracTrueStubsVsEtaMatched_[fitName]->Fill(fabs(s->eta()), trueStub);
	}

      } else {
	hisFitQinvPtUnmatched_[fitName]->Fill( fitTrk.qOverPt() );
	hisFitPhi0Unmatched_[fitName]->Fill( fitTrk.phi0() );
	hisFitD0Unmatched_[fitName]->Fill( fitTrk.d0() );
	hisFitZ0Unmatched_[fitName]->Fill( fitTrk.z0() );
	hisFitEtaUnmatched_[fitName]->Fill( fitTrk.eta() );
 
	hisFitChi2Unmatched_[fitName]->Fill( fitTrk.chi2() );
	hisFitChi2DofUnmatched_[fitName]->Fill( fitTrk.chi2dof() );
	hisFitChi2DofRphiUnmatched_[fitName]->Fill( fitTrk.chi2rphi() / fitTrk.numDOFrphi());
	hisFitChi2DofRzUnmatched_[fitName]->Fill( fitTrk.chi2rz() / fitTrk.numDOFrz());
	if (settings_->kalmanAddBeamConstr() && fitName.find("KF5") != string::npos) { // Histograms of chi2 with beam-spot constraint only make sense for 5 param fit.
	  hisFitBeamChi2Unmatched_[fitName]->Fill( fitTrk.chi2_bcon() );
	  hisFitBeamChi2DofUnmatched_[fitName]->Fill( fitTrk.chi2dof_bcon() );
	}
	profFitChi2VsEtaUnmatched_[fitName]->Fill( fabs(fitTrk.eta()), fitTrk.chi2() );
	profFitChi2DofVsEtaUnmatched_[fitName]->Fill( fabs(fitTrk.eta()), fitTrk.chi2dof() );
	profFitChi2VsInvPtUnmatched_[fitName]->Fill( fabs(fitTrk.qOverPt()), fitTrk.chi2() );
	profFitChi2DofVsInvPtUnmatched_[fitName]->Fill( fabs(fitTrk.qOverPt()), fitTrk.chi2dof() ); 

	if (fitName.find("KF") != string::npos) {
	  // No. of skipped layers on track during Kalman track fit.
	  if (nSkippedLayers == 0) {
	    hisKalmanChi2DofSkipLay0Unmatched_[fitName]->Fill(fitTrk.chi2dof());
	  } else if (nSkippedLayers == 1) {
	    hisKalmanChi2DofSkipLay1Unmatched_[fitName]->Fill(fitTrk.chi2dof());
	  } else if (nSkippedLayers >= 2) {
	    hisKalmanChi2DofSkipLay2Unmatched_[fitName]->Fill(fitTrk.chi2dof());
	  }
	}

	hisFitVsSeedQinvPtUnmatched_[fitName]->Fill( htTrk.qOverPt(), fitTrk.qOverPt() );
	hisFitVsSeedPhi0Unmatched_[fitName]->Fill( htTrk.phi0(), fitTrk.phi0() );
	hisFitVsSeedD0Unmatched_[fitName]->Fill( htTrk.d0(), fitTrk.d0() );
	hisFitVsSeedZ0Unmatched_[fitName]->Fill( htTrk.z0(), fitTrk.z0() );
	hisFitVsSeedEtaUnmatched_[fitName]->Fill( htTrk.eta(), fitTrk.eta() );
      }

      // Study how incorrect stubs on track affect fit chi2.
      profFitChi2VsPurity_[fitName]->Fill( fitTrk.getPurity(), fitTrk.chi2());
      profFitChi2DofVsPurity_[fitName]->Fill( fitTrk.getPurity(), fitTrk.chi2dof());

      // Look at stub residuals w.r.t. fitted (or true) track.
      if (tp != nullptr) {
	//if (tp != nullptr && tp->pt() > 50 && fabs(tp->pdgId()) == 13 && tp->charge() > 0) {
        //if (tp != nullptr && tp->pt() > 2 && tp->pt() < 2.5 && fabs(tp->pdgId()) == 13 && tp->charge() > 0) {
	// --- Options for recalc histograms
	// Choose to get residuals from truth particle or fitted track?
	// (Only applies to chi2 method 2 below).
	const bool recalc_useTP = false; 
	// debug printout
	const bool recalc_debug = false;
	// In residual plots, use residuals from method 2. (Imperfect, as neglects r-phi to r-z correlation).
	const bool recalc_method2 = true;

	float recalcChiSquared_1_rphi = 0.;
	float recalcChiSquared_1_rz = 0.;
	float recalcChiSquared_2 = 0.;
	const vector<const Stub*> stubs = fitTrk.getStubs();
	if (recalc_debug) cout<<"RECALC loop stubs : HT cell=("<<fitTrk.getCellLocationHT().first<<","<<fitTrk.getCellLocationHT().second<<")   TP PDG_ID="<<tp->pdgId()<<endl;
	for (const Stub* s : stubs) {
	  // Was this stub produced by correct truth particle?
	  const set<const TP*> stubTPs = s->assocTPs();
	  bool trueStub = (stubTPs.find(tp) != stubTPs.end());

	  //--- Calculation of chi2 (method 1 -- works with residuals in (z,phi) in barrel & (r,phi) in endcap & allows for non-radial 2S endcap strips by shifting stub phi coords.)

	  //--- Calculate residuals
	  // Distance of stub from true trajectory in z (barrel) or r (endcap)
	  float deltaRorZ =  s->barrel()  ?  (s->z() - tp->trkZAtStub( s ))  :  (s->r() - tp->trkRAtStub( s ));
	  // Distance of stub from true trajectory in phi.
	  float deltaPhi  = reco::deltaPhi(s->phi(), tp->trkPhiAtStub(s));

	  // Nasty correction to stub phi coordinate to take into account non-radial strips in endcap 2S modules.
	  float phiCorr = (tp->trkRAtStub(s) - s->r()) * s->alpha();
	  deltaPhi += phiCorr;

	  // Local calculation of chi2, to check that from fitter.
	  float sigmaPhi2_raw = pow(s->sigmaPerp() / s->r(), 2); 
	  float sigmaZ2_raw = s->sigmaPar()  * s->sigmaPar();
	  float sigmaZ2 = sigmaZ2_raw;
	  // Scattering term scaling as 1/Pt.
	  double phiExtra = settings_->kalmanMultiScattTerm()/(tp->pt());
	  double phiExtra2 = phiExtra * phiExtra;
	  float sigmaPhi2 = sigmaPhi2_raw + phiExtra2;
	  if (s->tiltedBarrel()) {
	    float tilt = s->moduleTilt();
	    float scaleTilted = sin(tilt) + cos(tilt)*(tp->tanLambda());
	    float scaleTilted2 = scaleTilted*scaleTilted;
	    sigmaZ2 *= scaleTilted2;
	  }
	  if (trueStub) {
	    recalcChiSquared_1_rphi += pow((deltaPhi), 2) / sigmaPhi2;
	    recalcChiSquared_1_rz += pow(deltaRorZ, 2) / sigmaZ2;
	  }

	  //--- Calculation of chi2 (method 2 -- works with residuals in (z,phi) everywhere & allows for non-radial 2S endcap strips via correlation matrix in stub coords - copied from KF4ParamsComb.cc)

	  //--- Calculate residuals
	  float deltaZ_proj, deltaPhi_proj;
	  float inv2R_proj, tanL_proj, z0_proj;
	  if (recalc_useTP) {
	    inv2R_proj = tp->qOverPt() * (0.5 * settings_->invPtToInvR());
	    tanL_proj  = tp->tanLambda();
	    z0_proj    = tp->z0();
	    // Distance of stub from true trajectory in z, evalulated at nominal radius of stub.
	    //deltaZ_proj =  s->z() - tp->trkZAtR(s->r());
	    deltaZ_proj =  s->z() - (tp->z0() + tp->tanLambda() * s->r());
	    // Distance of stub from true trajectory in r*phi, evaluated at nominal radius of stub.
	    //deltaPhi_proj  = reco::deltaPhi(s->phi(), tp->trkPhiAtR(s->r()));
	    deltaPhi_proj  = reco::deltaPhi(s->phi(), tp->phi0() - (s->r() * inv2R_proj));
	  } else {
	    inv2R_proj = fitTrk.qOverPt() * (0.5 * settings_->invPtToInvR());
	    tanL_proj  = fitTrk.tanLambda();
	    z0_proj    = fitTrk.z0();
	    deltaZ_proj =  s->z() - (fitTrk.z0() + fitTrk.tanLambda() * s->r());
	    deltaPhi_proj  = reco::deltaPhi(s->phi(), fitTrk.phi0() - (s->r() * inv2R_proj));
	  }

	  // Higher order correction correction to circle expansion for improved accuracy at low Pt.
	  float corr = s->r() * inv2R_proj; // = r/2R
	  deltaPhi_proj += (1./6.)*pow(corr, 3);

	  if ( (not s->barrel()) && not (s->psModule())) {
	    // These corrections rely on inside --> outside tracking, so r-z track params in 2S modules known.
	    float rShift = (s->z() - z0_proj)/tanL_proj - s->r();

	    if (settings_->kalmanHOprojZcorr() == 1) {
	      // Add correlation term related to conversion of stub residuals from (r,phi) to (z,phi).
	      deltaPhi_proj += inv2R_proj * rShift;
	    }

	    if (settings_->kalmanHOalpha()     == 1) {
	      // Add alpha correction for non-radial 2S endcap strips..
	      deltaPhi_proj += s->alpha() * rShift;
	    }
	  }

	  float sigmaZ2_proj   = (s->barrel())  ?  sigmaZ2    :  sigmaZ2*pow(tp->tanLambda(), 2);
	  float beta = 0.;
	  if (not s->barrel()) {
	    // Add correlation term related to conversion of stub residuals from (r,phi) to (z,phi).
	    if (settings_->kalmanHOprojZcorr() == 2) beta += -inv2R_proj;
	    // Add alpha correction for non-radial 2S endcap strips..
	    if (settings_->kalmanHOalpha()     == 2) beta += -s->alpha(); 
	  }
	  float beta2 = beta*beta;
	  float sigmaPhi2_proj = sigmaPhi2_raw + sigmaZ2_raw * beta2;
	  // Add scatterign uncertainty
	  if (recalc_useTP) {
	    sigmaPhi2_proj += phiExtra2;
	  } else {
	    // Fit uses Pt of L1track3D to estimate scattering.
	    double phiExtra_fit = settings_->kalmanMultiScattTerm()/(fitTrk.getL1track3D().pt());
	    double phiExtra2_fit = phiExtra_fit * phiExtra_fit;
	    sigmaPhi2_proj += phiExtra2_fit;
	  }
	  // Correlation of stub phi & stub z coordinates.
	  float sigmaCorr = (s->barrel())  ?   0.0  :  sigmaZ2_raw * beta * tp->tanLambda();

	  // Invert covariance matrix in stub position uncertainty.
	  float det = sigmaPhi2_proj * sigmaZ2_proj - sigmaCorr * sigmaCorr; 
	  float V00 = sigmaZ2_proj / det;
	  float V01 = -sigmaCorr / det;
	  float V11 = sigmaPhi2_proj / det;
	  if (trueStub) {
	    recalcChiSquared_2 += V00 * pow(deltaPhi_proj, 2) + V11 * pow(deltaZ_proj, 2) + 2 * V01 * deltaPhi_proj * deltaZ_proj;
	    if (recalc_debug) {
	      cout<<"  RECALC BARREL="<<s->barrel()<<" PS="<<s->psModule()<<" ID="<<s->index()<<endl;
	      cout<<"  RECALC RESID: 1000*rphi="<<1000*deltaPhi_proj<<" rz="<<deltaZ_proj<<endl;
	      cout<<"  RECALC SIGMA: 1000*rphi="<<1000*sqrt(sigmaPhi2_proj)<<" rz="<<sqrt(sigmaZ2_proj)<<endl;
	      cout<<"  RECALC CHI2="<<recalcChiSquared_2<<" & DELTA CHI2: rphi="<<V00 * pow(deltaPhi_proj, 2)<<" rz="<<V11 * pow(deltaZ_proj, 2)<<endl<<endl;
	    }
	  }

    float sigmaPhi = sqrt(sigmaPhi2);
    float sigmaZ    = sqrt(sigmaZ2);

	  if (recalc_method2) {
	    // Plot residuals from method 2.
	    // (Neglects effect of correlation term sigmaCorr).
	    deltaPhi  = deltaPhi_proj;
	    deltaRorZ = deltaZ_proj;
	    sigmaPhi  = sqrt(sigmaPhi2_proj);
	    sigmaZ    = sqrt(sigmaZ2_proj);
	  }

	  if (trueStub) {
	    if (s->psModule()) {
	      if (s->barrel()) {
		hisDeltaPhitruePSbarrel_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZtruePSbarrel_[fitName]->Fill(deltaRorZ/sigmaZ);
	      } else{
		hisDeltaPhitruePSendcap_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZtruePSendcap_[fitName]->Fill(deltaRorZ/sigmaZ);
	      }
	    } else {
	      if (s->barrel()) {
		hisDeltaPhitrue2Sbarrel_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZtrue2Sbarrel_[fitName]->Fill(deltaRorZ/sigmaZ);
	      } else {
		hisDeltaPhitrue2Sendcap_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZtrue2Sendcap_[fitName]->Fill(deltaRorZ/sigmaZ);
	      }
	    }
	    // More detailed plots for true stubs to study effect of multiple scattering.
	    profNsigmaPhivsInvPt_[fitName]->Fill(1./tp->pt(), fabs(deltaPhi/sigmaPhi));
	    profNsigmaPhivsR_[fitName]->Fill(s->r(),   fabs(deltaPhi/sigmaPhi));
	    profNsigmaPhivsTanl_[fitName]->Fill(fabs(tp->tanLambda()),   fabs(deltaPhi/sigmaPhi));
	  } else {
	    if (s->psModule()) {
	      if (s->barrel()) {
		hisDeltaPhifakePSbarrel_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZfakePSbarrel_[fitName]->Fill(deltaRorZ/sigmaZ);
	      } else {
		hisDeltaPhifakePSendcap_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZfakePSendcap_[fitName]->Fill(deltaRorZ/sigmaZ);
	      }
	    } else {
	      if (s->barrel()) {
		hisDeltaPhifake2Sbarrel_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZfake2Sbarrel_[fitName]->Fill(deltaRorZ/sigmaZ);
	      } else {
		hisDeltaPhifake2Sendcap_[fitName]->Fill(deltaPhi/sigmaPhi);
		hisDeltaRorZfake2Sendcap_[fitName]->Fill(deltaRorZ/sigmaZ);
	      }
	    }
	  }
	}
	// Plot recalculated chi2 for correct stubs on matched tracks.
	profRecalcRphiChi2VsEtaTrue1_[fitName]->Fill(fabs(fitTrk.eta()), recalcChiSquared_1_rphi);
	profRecalcRzChi2VsEtaTrue1_[fitName]->Fill(fabs(fitTrk.eta()), recalcChiSquared_1_rz);
	float recalcChiSquared_1 = recalcChiSquared_1_rphi + recalcChiSquared_1_rz;
	profRecalcChi2VsEtaTrue1_[fitName]->Fill(fabs(fitTrk.eta()), recalcChiSquared_1);
	profRecalcChi2VsEtaTrue2_[fitName]->Fill(fabs(fitTrk.eta()), recalcChiSquared_2);
      }
    }

    // Study helix param resolution.

    for (const L1fittedTrack& fitTrk : fittedTracks) {
      const TP* tp   = fitTrk.getMatchedTP();
      if ( tp != nullptr ){
	// IRT
	if ((resPlotOpt_ && tp->useForAlgEff()) || (not resPlotOpt_)) { // Check TP is good for efficiency measurement (& also comes from signal event if requested)
	  //if (not (abs(tp->pdgId()) == 11)) continue;
	  //if (not (abs(tp->pdgId()) == 13) || (abs(tp->pdgId()) == 211) || (abs(tp->pdgId()) == 321) || (abs(tp->pdgId()) == 2212)) continue;
	  // Fitted vs True parameter distribution 2D plots
	  hisFitVsTrueQinvPt_[fitName]->Fill( tp->qOverPt(), fitTrk.qOverPt() );
	  hisFitVsTruePhi0_[fitName]->Fill( tp->phi0(), fitTrk.phi0( ));
	  hisFitVsTrueD0_[fitName]->Fill( tp->d0(), fitTrk.d0() );
	  hisFitVsTrueZ0_[fitName]->Fill( tp->z0(), fitTrk.z0() );
	  hisFitVsTrueEta_[fitName]->Fill( tp->eta(), fitTrk.eta() );

	  // Residuals between fitted and true helix params as 1D plot.
	  hisFitQinvPtRes_[fitName]->Fill( fitTrk.qOverPt() - tp->qOverPt());
	  hisFitPhi0Res_[fitName]->Fill( reco::deltaPhi(fitTrk.phi0(), tp->phi0()) );
	  hisFitD0Res_[fitName]->Fill( fitTrk.d0() - tp->d0() );
	  hisFitZ0Res_[fitName]->Fill( fitTrk.z0() - tp->z0() );
	  hisFitEtaRes_[fitName]->Fill( fitTrk.eta() - tp->eta() );  

	  // Plot helix parameter resolution against eta or Pt.
	  hisQoverPtResVsTrueEta_[fitName]->Fill( std::abs(tp->eta()), std::abs( fitTrk.qOverPt() - tp->qOverPt() ) );
	  hisPhi0ResVsTrueEta_[fitName]->Fill( std::abs(tp->eta()), std::abs(reco::deltaPhi(fitTrk.phi0(), tp->phi0()) ) );
	  hisEtaResVsTrueEta_[fitName]->Fill( std::abs(tp->eta()), std::abs( fitTrk.eta() - tp->eta() ) );
	  hisZ0ResVsTrueEta_[fitName]->Fill( std::abs(tp->eta()), std::abs( fitTrk.z0() - tp->z0() ) );
	  hisD0ResVsTrueEta_[fitName]->Fill( std::abs(tp->eta()), std::abs( fitTrk.d0() - tp->d0() ) );

	  hisQoverPtResVsTrueInvPt_[fitName]->Fill( std::abs(tp->qOverPt()), std::abs( fitTrk.qOverPt() - tp->qOverPt() ) );
	  hisPhi0ResVsTrueInvPt_[fitName]->Fill( std::abs(tp->qOverPt()), std::abs(reco::deltaPhi(fitTrk.phi0(), tp->phi0()) ) );
	  hisEtaResVsTrueInvPt_[fitName]->Fill( std::abs(tp->qOverPt()), std::abs( fitTrk.eta() - tp->eta() ) );
	  hisZ0ResVsTrueInvPt_[fitName]->Fill( std::abs(tp->qOverPt()), std::abs( fitTrk.z0() - tp->z0() ) );
	  hisD0ResVsTrueInvPt_[fitName]->Fill( std::abs(tp->qOverPt()), std::abs( fitTrk.d0() - tp->d0() ) );

	  // Also plot resolution for 5 parameter fits after beam-spot constraint it applied post-fit.
	  if (settings_->kalmanAddBeamConstr() && fitName.find("KF5") != string::npos) {
	    hisQoverPtResBeamVsTrueEta_[fitName]->Fill( std::abs(tp->eta()), std::abs( fitTrk.qOverPt_bcon() - tp->qOverPt() ) );
	    hisPhi0ResBeamVsTrueEta_[fitName]->Fill( std::abs(tp->eta()), std::abs(reco::deltaPhi(fitTrk.phi0_bcon(), tp->phi0()) ) );

	    hisQoverPtResBeamVsTrueInvPt_[fitName]->Fill( std::abs(tp->qOverPt()), std::abs( fitTrk.qOverPt_bcon() - tp->qOverPt() ) );
	    hisPhi0ResBeamVsTrueInvPt_[fitName]->Fill( std::abs(tp->qOverPt()), std::abs(reco::deltaPhi(fitTrk.phi0_bcon(), tp->phi0()) ) );
	  }
	}
      }
    }

    //=== Study duplicate tracks.

    for (const TP& tp: vTPs) {
      if (tpRecoedMap[&tp]) { // Was this truth particle successfully fitted?
	profDupFitTrksVsEta_[fitName]->Fill(fabs(tp.eta()), tpRecoedDup[&tp] - 1); 
	profDupFitTrksVsInvPt_[fitName]->Fill(fabs(tp.qOverPt()), tpRecoedDup[&tp] - 1); 
      }
    }

    //=== Study tracking efficiency by looping over tracking particles.

    for (const TP& tp: vTPs) {

      if (tp.useForEff()) { // Check TP is good for efficiency measurement.

	// Check which phi & eta sectors this TP is in.
	int iEtaReg_TP = -1;
	int iPhiSec_TP = -1;
	for (unsigned int iPhiSec = 0; iPhiSec < numPhiSectors_; iPhiSec++) {
	  Sector secTmp;
	  secTmp.init(settings_, iPhiSec, 0);
	  if (secTmp.insidePhiSec(tp)) iPhiSec_TP = iPhiSec;
	}
	for (unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++) {
	  Sector secTmp;
	  secTmp.init(settings_, 0, iEtaReg);
	  if (secTmp.insideEtaReg(tp)) iEtaReg_TP = iEtaReg;
	}

	// If TP was reconstucted by HT, then plot its kinematics.
	if (tpRecoedMap[&tp]) {  // This truth particle was successfully fitted.
	  hisFitTPinvptForEff_[fitName]->Fill(1./tp.pt());
	  hisFitTPptForEff_[fitName]->Fill(tp.pt());
	  hisFitTPetaForEff_[fitName]->Fill(tp.eta());
	  hisFitTPphiForEff_[fitName]->Fill(tp.phi0());
	  // Plot also production point of all good reconstructed TP.
	  hisFitTPd0ForEff_[fitName]->Fill(fabs(tp.d0()));
	  hisFitTPz0ForEff_[fitName]->Fill(fabs(tp.z0()));
	  // Also plot efficiency to perfectly reconstruct the track (no fake hits)
	  if (tpPerfRecoedMap[&tp]) { // This truth particle was successfully fitted with no incorrect hits.
	    hisPerfFitTPinvptForEff_[fitName]->Fill(1./tp.pt());
	    hisPerfFitTPptForEff_[fitName]->Fill(tp.pt());
	    hisPerfFitTPetaForEff_[fitName]->Fill(tp.eta());
	  }
	  if (tp.useForAlgEff()) { // Check TP is good for algorithmic efficiency measurement.
	    hisFitTPinvptForAlgEff_[fitName]->Fill(1./tp.pt());
	    hisFitTPptForAlgEff_[fitName]->Fill(tp.pt());
	    hisFitTPetaForAlgEff_[fitName]->Fill(tp.eta());
	    hisFitTPphiForAlgEff_[fitName]->Fill(tp.phi0());
	    // Plot also production point of all good reconstructed TP.
	    hisFitTPd0ForAlgEff_[fitName]->Fill(fabs(tp.d0()));
	    hisFitTPz0ForAlgEff_[fitName]->Fill(fabs(tp.z0()));
	    // Plot sector number to understand if looser cuts are needed in certain regions.
	    hisFitTPphisecForAlgEff_[fitName]->Fill(iPhiSec_TP);
	    hisFitTPetasecForAlgEff_[fitName]->Fill(iEtaReg_TP);
	    // Also plot efficiency to perfectly reconstruct the track (no fake hits)
	    if (tpPerfRecoedMap[&tp]) {
	      hisPerfFitTPinvptForAlgEff_[fitName]->Fill(1./tp.pt());
	      hisPerfFitTPptForAlgEff_[fitName]->Fill(tp.pt());
	      hisPerfFitTPetaForAlgEff_[fitName]->Fill(tp.eta());
	      hisPerfFitTPphisecForAlgEff_[fitName]->Fill(iPhiSec_TP);
	      hisPerfFitTPetasecForAlgEff_[fitName]->Fill(iEtaReg_TP);
	      // Efficiency inside jets.
	      if ( tp.tpInJet() ) {
		hisPerfFitTPinvptForAlgEff_inJetPtG30_[fitName]->Fill(1./tp.pt());
	      }
	      if ( tp.tpInHighPtJet() ) {
		hisPerfFitTPinvptForAlgEff_inJetPtG100_[fitName]->Fill(1./tp.pt());           
	      }
	      if ( tp.tpInVeryHighPtJet() ) {
		hisPerfFitTPinvptForAlgEff_inJetPtG200_[fitName]->Fill(1./tp.pt());           
	      }
	    }
	  }
	}
      }
    }
  }
}

//=== Produce plots of tracking efficiency after HT or after r-z track filter (run at end of job).

TFileDirectory Histos::plotTrackEfficiency(string tName) {

  // Define lambda function to facilitate adding "tName" to directory & histogram names.
  auto addn = [tName](string s){ return TString::Format("%s_%s", s.c_str(), tName.c_str()); };

  TFileDirectory inputDir = fs_->mkdir(addn("Effi").Data());
  // Plot tracking efficiency
  makeEfficiencyPlot(inputDir, teffEffVsInvPt_[tName], hisRecoTPinvptForEff_[tName], hisTPinvptForEff_,
		     addn("EffVsInvPt"), "; 1/Pt; Tracking efficiency" );
  makeEfficiencyPlot(inputDir, teffEffVsPt_[tName], hisRecoTPptForEff_[tName], hisTPptForEff_,
		     addn("EffVsPt"), "; Pt; Tracking efficiency" );
  makeEfficiencyPlot(inputDir, teffEffVsEta_[tName], hisRecoTPetaForEff_[tName], hisTPetaForEff_,
		     addn("EffVsEta"), "; #eta; Tracking efficiency" );

  // std::cout << "Made first graph" << std::endl;
  // graphEffVsEta_[tName] = inputDir.make<TGraphAsymmErrors>(hisRecoTPetaForEff_[tName], hisTPetaForEff_);
  // graphEffVsEta_[tName]->SetNameTitle("EffVsEta","; #eta; Tracking efficiency");
  makeEfficiencyPlot(inputDir, teffEffVsPhi_[tName], hisRecoTPphiForEff_[tName], hisTPphiForEff_,
		     addn("EffVsPhi"), "; #phi; Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffEffVsD0_[tName], hisRecoTPd0ForEff_[tName], hisTPd0ForEff_, 
		      addn("EffVsD0"),"; d0 (cm); Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffEffVsZ0_[tName], hisRecoTPz0ForEff_[tName], hisTPz0ForEff_, 
		      addn("EffVsZ0"),"; z0 (cm); Tracking efficiency");

  // Also plot efficiency to reconstruct track perfectly.
  makeEfficiencyPlot( inputDir, teffPerfEffVsInvPt_[tName], hisPerfRecoTPinvptForEff_[tName], hisTPinvptForEff_, 
		      addn("PerfEffVsInvPt"),"; 1/Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfEffVsPt_[tName], hisPerfRecoTPptForEff_[tName], hisTPptForEff_, 
		      addn("PerfEffVsPt"),"; Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfEffVsEta_[tName], hisPerfRecoTPetaForEff_[tName], hisTPetaForEff_, 
		      addn("PerfEffVsEta"),"; #eta; Tracking perfect efficiency");

  // Plot algorithmic tracking efficiency
  makeEfficiencyPlot( inputDir, teffAlgEffVsInvPt_[tName], hisRecoTPinvptForAlgEff_[tName], hisTPinvptForAlgEff_,
		      addn("AlgEffVsInvPt"),"; 1/Pt; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffVsPt_[tName], hisRecoTPptForAlgEff_[tName], hisTPptForAlgEff_,
		      addn("AlgEffVsPt"),"; Pt; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffVsEta_[tName], hisRecoTPetaForAlgEff_[tName], hisTPetaForAlgEff_, 
		      addn("AlgEffVsEta"),"; #eta; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffVsPhi_[tName], hisRecoTPphiForAlgEff_[tName], hisTPphiForAlgEff_, 
		      addn("AlgEffVsPhi"),"; #phi; Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffAlgEffVsD0_[tName], hisRecoTPd0ForAlgEff_[tName], hisTPd0ForAlgEff_, 
		      addn("AlgEffVsD0"),"; d0 (cm); Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffVsZ0_[tName], hisRecoTPz0ForAlgEff_[tName], hisTPz0ForAlgEff_, 
		      addn("AlgEffVsZ0"),"; z0 (cm); Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffAlgEffVsPhiSec_[tName], hisRecoTPphisecForAlgEff_[tName], hisTPphisecForAlgEff_,
		      addn("AlgEffVsPhiSec"),"; #phi sector; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffVsEtaSec_[tName], hisRecoTPetasecForAlgEff_[tName], hisTPetasecForAlgEff_,
		      addn("AlgEffVsEtaSec"),"; #eta sector; Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffAlgEffVsInvPt_inJetPtG30_[tName], hisRecoTPinvptForAlgEff_inJetPtG30_[tName], hisTPinvptForAlgEff_inJetPtG30_,
          addn("AlgEffVsInvPt_inJetPtG30"),"; 1/Pt; Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffAlgEffVsInvPt_inJetPtG100_[tName], hisRecoTPinvptForAlgEff_inJetPtG100_[tName], hisTPinvptForAlgEff_inJetPtG100_,
          addn("AlgEffVsInvPt_inJetPtG100"),"; 1/Pt; Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffAlgEffVsInvPt_inJetPtG200_[tName], hisRecoTPinvptForAlgEff_inJetPtG200_[tName], hisTPinvptForAlgEff_inJetPtG200_,
          addn("AlgEffVsInvPt_inJetPtG200"),"; 1/Pt; Tracking efficiency");

  // Also plot algorithmic efficiency to reconstruct track perfectly.
  makeEfficiencyPlot( inputDir, teffPerfAlgEffVsInvPt_[tName], hisPerfRecoTPinvptForAlgEff_[tName], hisTPinvptForAlgEff_,
		      addn("PerfAlgEffVsInvPt"),"; 1/Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffVsPt_[tName], hisPerfRecoTPptForAlgEff_[tName], hisTPptForAlgEff_,
		      addn("PerfAlgEffVsPt"),"; Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffVsEta_[tName], hisPerfRecoTPetaForAlgEff_[tName], hisTPetaForAlgEff_,
		      addn("PerfAlgEffVsEta"),"; #eta; Tracking perfect efficiency");

  makeEfficiencyPlot( inputDir, teffPerfAlgEffVsPhiSec_[tName], hisPerfRecoTPphisecForAlgEff_[tName], hisTPphisecForAlgEff_,
		      addn("PerfAlgEffVsPhiSec"),"; #phi sector; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffVsEtaSec_[tName], hisPerfRecoTPetasecForAlgEff_[tName], hisTPetasecForAlgEff_,
		      addn("PerfAlgEffVsEtaSec"),"; #eta sector; Tracking perfect efficiency");

  return inputDir;
}

//=== Produce plots of tracking efficiency after track fit (run at end of job).

TFileDirectory Histos::plotTrackEffAfterFit(string fitName) {

  // Define lambda function to facilitate adding "fitName" to directory & histogram names.
  auto addn = [fitName](string s){ return TString::Format("%s_%s", s.c_str(), fitName.c_str()); };

  TFileDirectory inputDir = fs_->mkdir(addn("Effi").Data());
  // Plot tracking efficiency
  makeEfficiencyPlot( inputDir, teffEffFitVsInvPt_[fitName], hisFitTPinvptForEff_[fitName], hisTPinvptForEff_, 
		      addn("EffFitVsInvPt"),"; 1/Pt; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffEffFitVsPt_[fitName], hisFitTPptForEff_[fitName], hisTPptForEff_, 
		      addn("EffFitVsPt"),"; Pt; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffEffFitVsEta_[fitName], hisFitTPetaForEff_[fitName], hisTPetaForEff_, 
		      addn("EffFitVsEta"),"; #eta; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffEffFitVsPhi_[fitName], hisFitTPphiForEff_[fitName], hisTPphiForEff_, 
		      addn("EffFitVsPhi"),"; #phi; Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffEffFitVsD0_[fitName], hisFitTPd0ForEff_[fitName], hisTPd0ForEff_, 
		      addn("EffFitVsD0"),"; d0 (cm); Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffEffFitVsZ0_[fitName], hisFitTPz0ForEff_[fitName], hisTPz0ForEff_, 
		      addn("EffFitVsZ0"),"; z0 (cm); Tracking efficiency");

  // Also plot efficiency to reconstruct track perfectly.
  makeEfficiencyPlot( inputDir, teffPerfEffFitVsInvPt_[fitName], hisPerfFitTPinvptForEff_[fitName], hisTPinvptForEff_,
		      addn("PerfEffFitVsInvPt"),"; 1/Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfEffFitVsPt_[fitName], hisPerfFitTPptForEff_[fitName], hisTPptForEff_,
		      addn("PerfEffFitVsPt"),"; Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfEffFitVsEta_[fitName], hisPerfFitTPetaForEff_[fitName], hisTPetaForEff_, 
		      addn("PerfEffFitVsEta"),"; #eta; Tracking perfect efficiency");

  // Plot algorithmic tracking efficiency
  makeEfficiencyPlot( inputDir, teffAlgEffFitVsInvPt_[fitName], hisFitTPinvptForAlgEff_[fitName], hisTPinvptForAlgEff_,
		      addn("AlgEffFitVsInvPt"),"; 1/Pt; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffFitVsPt_[fitName], hisFitTPptForAlgEff_[fitName], hisTPptForAlgEff_,
		      addn("AlgEffFitVsPt"),"; Pt; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffFitVsEta_[fitName], hisFitTPetaForAlgEff_[fitName], hisTPetaForAlgEff_, 
		      addn("AlgEffFitVsEta"),"; #eta; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffFitVsPhi_[fitName], hisFitTPphiForAlgEff_[fitName], hisTPphiForAlgEff_, 
		      addn("AlgEffFitVsPhi"),"; #phi; Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffAlgEffFitVsD0_[fitName], hisFitTPd0ForAlgEff_[fitName], hisTPd0ForAlgEff_, 
		      addn("AlgEffFitVsD0"),"; d0 (cm); Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffFitVsZ0_[fitName], hisFitTPz0ForAlgEff_[fitName], hisTPz0ForAlgEff_, 
		      addn("AlgEffFitVsZ0"),"; z0 (cm); Tracking efficiency");

  makeEfficiencyPlot( inputDir, teffAlgEffFitVsPhiSec_[fitName], hisFitTPphisecForAlgEff_[fitName], hisTPphisecForAlgEff_,
		      addn("AlgEffFitVsPhiSec"),"; #phi sector; Tracking efficiency");
  makeEfficiencyPlot( inputDir, teffAlgEffFitVsEtaSec_[fitName], hisFitTPetasecForAlgEff_[fitName], hisTPetasecForAlgEff_,
		      addn("AlgEffFitVsEtaSec"),"; #eta sector; Tracking efficiency");

  // Also plot algorithmic efficiency to reconstruct track perfectly.
  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsInvPt_[fitName], hisPerfFitTPinvptForAlgEff_[fitName], hisTPinvptForAlgEff_,
		      addn("PerfAlgEffFitVsInvPt"),"; 1/Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsPt_[fitName], hisPerfFitTPptForAlgEff_[fitName], hisTPptForAlgEff_,
		      addn("PerfAlgEffFitVsPt"),"; Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsEta_[fitName], hisPerfFitTPetaForAlgEff_[fitName], hisTPetaForAlgEff_, 
		      addn("Perf AlgEffFitVsEta"),"; #eta; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsInvPt_inJetPtG30_[fitName], hisPerfFitTPinvptForAlgEff_inJetPtG30_[fitName], hisTPinvptForAlgEff_inJetPtG30_,
		      addn("PerfAlgEffFitVsInvPt_inJetPtG30"),"; 1/Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsInvPt_inJetPtG100_[fitName], hisPerfFitTPinvptForAlgEff_inJetPtG100_[fitName], hisTPinvptForAlgEff_inJetPtG100_,
		      addn("PerfAlgEffFitVsInvPt_inJetPtG100"),"; 1/Pt; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsInvPt_inJetPtG200_[fitName], hisPerfFitTPinvptForAlgEff_inJetPtG200_[fitName], hisTPinvptForAlgEff_inJetPtG200_,
		      addn("PerfAlgEffFitVsInvPt_inJetPtG200"),"; 1/Pt; Tracking perfect efficiency");

  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsPhiSec_[fitName], hisPerfFitTPphisecForAlgEff_[fitName], hisTPphisecForAlgEff_,
		      addn("PerfAlgEffFitVsPhiSec"),"; #phi sector; Tracking perfect efficiency");
  makeEfficiencyPlot( inputDir, teffPerfAlgEffFitVsEtaSec_[fitName], hisPerfFitTPetasecForAlgEff_[fitName], hisTPetasecForAlgEff_,
		      addn("PerfAlgEffFitVsEtaSec"),"; #eta sector; Tracking perfect efficiency");

  return inputDir;
}

void Histos::makeEfficiencyPlot( TFileDirectory &inputDir, TEfficiency* outputEfficiency, TH1F* pass, TH1F* all, TString name, TString title ) {

  outputEfficiency = inputDir.make<TEfficiency>(*pass, *all);
  outputEfficiency->SetName(name);
  outputEfficiency->SetTitle(title);
}

//=== Print summary of track-finding performance after track pattern reco.

void Histos::printTrackPerformance(string tName) {

  float numTrackCands = profNumTrackCands_[tName]->GetBinContent(1); // No. of track cands
  float numTrackCandsErr = profNumTrackCands_[tName]->GetBinError(1); // No. of track cands uncertainty
  float numMatchedTrackCandsIncDups = profNumTrackCands_[tName]->GetBinContent(2); // Ditto, counting only those matched to TP
  float numMatchedTrackCandsExcDups = profNumTrackCands_[tName]->GetBinContent(6); // Ditto, but excluding duplicates
  float numFakeTracks = numTrackCands - numMatchedTrackCandsIncDups;
  float numExtraDupTracks = numMatchedTrackCandsIncDups - numMatchedTrackCandsExcDups;
  float fracFake = numFakeTracks/(numTrackCands + 1.0e-6);
  float fracDup = numExtraDupTracks/(numTrackCands + 1.0e-6);

  float numStubsOnTracks = profStubsOnTracks_[tName]->GetBinContent(1);
  float meanStubsPerTrack = numStubsOnTracks/(numTrackCands + 1.0e-6); //protection against demoninator equals zero.
  unsigned int numRecoTPforAlg = hisRecoTPinvptForAlgEff_[tName]->GetEntries();
  // Histograms of input truth particles (e.g. hisTPinvptForAlgEff_), used for denominator of efficiencies, are identical, 
  // irrespective of whether made after HT or after r-z track filter, so always use the former.
  unsigned int numTPforAlg     = hisTPinvptForAlgEff_->GetEntries();
  unsigned int numPerfRecoTPforAlg = hisPerfRecoTPinvptForAlgEff_[tName]->GetEntries();
  float algEff = float(numRecoTPforAlg)/(numTPforAlg + 1.0e-6); //protection against demoninator equals zero.
  float algEffErr = sqrt(algEff*(1-algEff)/(numTPforAlg + 1.0e-6)); // uncertainty
  float algPerfEff = float(numPerfRecoTPforAlg)/(numTPforAlg + 1.0e-6); //protection against demoninator equals zero.
  float algPerfEffErr = sqrt(algPerfEff*(1-algPerfEff)/(numTPforAlg + 1.0e-6)); // uncertainty

  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(4);

  cout<<"========================================================================="<<endl;
  if (tName == "HT") {
    cout<<"               TRACK-FINDING SUMMARY AFTER HOUGH TRANSFORM             "<<endl;
  } else if (tName == "RZ") {
    cout<<"               TRACK-FINDING SUMMARY AFTER R-Z TRACK FILTER            "<<endl;
  } else if (tName == "TRACKLET") {
    cout<<"               TRACK-FINDING SUMMARY AFTER TRACKLET PATTERN RECO       "<<endl;
  }
  cout<<"Number of track candidates found per event = "<<numTrackCands<<" +- "<<numTrackCandsErr<<endl;
  cout<<"                     with mean stubs/track = "<<meanStubsPerTrack<<endl; 
  cout<<"Fraction of track cands that are fake = "<<fracFake<<endl;
  cout<<"Fraction of track cands that are genuine, but extra duplicates = "<<fracDup<<endl;
  cout<<"Algorithmic tracking efficiency = "<<numRecoTPforAlg<<"/"<<numTPforAlg<<" = "<<algEff<<" +- "<<algEffErr<<endl;
  cout<<"Perfect algorithmic tracking efficiency = "<<numPerfRecoTPforAlg<<"/"<<numTPforAlg<<" = "<<algPerfEff<<" +- "<<algPerfEffErr<<" (no incorrect hits)"<<endl;
}

//=== Print summary of track-finding performance after helix fit for given track fitter.

void Histos::printFitTrackPerformance(string fitName) {

  float numFitTracks = profNumFitTracks_[fitName]->GetBinContent(1); // No. of track cands
  float numFitTracksErr = profNumFitTracks_[fitName]->GetBinError(1); // No. of track cands uncertainty
  float numMatchedFitTracksIncDups = profNumFitTracks_[fitName]->GetBinContent(2); // Ditto, counting only those matched to TP
  float numMatchedFitTracksExcDups = profNumFitTracks_[fitName]->GetBinContent(6); // Ditto, but excluding duplicates
  float numFakeFitTracks = numFitTracks - numMatchedFitTracksIncDups;
  float numExtraDupFitTracks = numMatchedFitTracksIncDups - numMatchedFitTracksExcDups;
  float fracFakeFit = numFakeFitTracks/(numFitTracks + 1.0e-6);
  float fracDupFit = numExtraDupFitTracks/(numFitTracks + 1.0e-6);

  float numStubsOnFitTracks = profStubsOnFitTracks_[fitName]->GetBinContent(1);
  float meanStubsPerFitTrack = numStubsOnFitTracks/(numFitTracks + 1.0e-6); //protection against demoninator equals zero.
  unsigned int numFitTPforAlg = hisFitTPinvptForAlgEff_[fitName]->GetEntries();
  // Histograms of input truth particles (e.g. hisTPinvptForAlgEff_), used for denominator of efficiencies, are identical, 
  // irrespective of whether made after HT or after r-z track filter, so always use the former.
  unsigned int numTPforAlg     = hisTPinvptForAlgEff_->GetEntries();
  unsigned int numPerfFitTPforAlg = hisPerfFitTPinvptForAlgEff_[fitName]->GetEntries();
  float fitEff = float(numFitTPforAlg)/(numTPforAlg + 1.0e-6); //protection against demoninator equals zero.
  float fitEffErr = sqrt(fitEff*(1-fitEff)/(numTPforAlg + 1.0e-6)); // uncertainty
  float fitPerfEff = float(numPerfFitTPforAlg)/(numTPforAlg + 1.0e-6); //protection against demoninator equals zero.
  float fitPerfEffErr = sqrt(fitPerfEff*(1-fitPerfEff)/(numTPforAlg + 1.0e-6)); // uncertainty

  // Does this fitter require r-z track filter to be run before it?
  bool useRZfilt = (std::count(useRZfilter_.begin(), useRZfilter_.end(), fitName) > 0);

  cout<<"========================================================================="<<endl;
  cout << "                    TRACK FIT SUMMARY FOR: " << fitName << endl;
  cout<<"Number of fitted track candidates found per event = "<<numFitTracks<<" +- "<<numFitTracksErr<<endl;
  cout<<"                     with mean stubs/track = "<<meanStubsPerFitTrack<<endl; 
  cout<<"Fraction of fitted tracks that are fake = "<<fracFakeFit<<endl;
  cout<<"Fraction of fitted tracks that are genuine, but extra duplicates = "<<fracDupFit<<endl;
  cout<<"Algorithmic fitting efficiency = "<<numFitTPforAlg<<"/"<<numTPforAlg<<" = "<<fitEff<<" +- "<<fitEffErr<<endl;
  cout<<"Perfect algorithmic fitting efficiency = "<<numPerfFitTPforAlg<<"/"<<numTPforAlg<<" = "<<fitPerfEff<<" +- "<<fitPerfEffErr<<" (no incorrect hits)"<<endl;
  if (useRZfilt) cout<<"(The above fitter used the '"<<settings_->rzFilterName()<<"' r-z track filter.)"<<endl;

  /*
    if ( settings_->detailedFitOutput() ){
    cout << endl<< "More detailed information about helix fit:"<<endl<<endl;
    }
  */
}

//=== Print tracking performance summary & make tracking efficiency histograms.

void Histos::endJobAnalysis() {

  // Don't bother producing summary if user didn't request histograms via TFileService in their cfg.
  if ( ! this->enabled() ) return;

  // Protection when running in wierd mixed hybrid-TMTT modes.
  bool wierdMixedMode = (hisRecoTPinvptForEff_.find("TRACKLET") == hisRecoTPinvptForEff_.end());

  if (settings_->hybrid() && not wierdMixedMode) {

    // Produce plots of tracking efficieny after tracklet pattern reco.
    this->plotTrackletSeedEfficiency();
    this->plotTrackEfficiency("TRACKLET");
    this->plotHybridDupRemovalEfficiency();

  } else {

    // Produce plots of tracking efficiency using track candidates found after HT.
    this->plotTrackEfficiency("HT");

    // Optionally produce plots of tracking efficiency using track candidates found after r-z track filter.
    if (ranRZfilter_) this->plotTrackEfficiency("RZ");
  }

  // Produce more plots of tracking efficiency using track candidates after track fit.
  for (auto &fitName : trackFitters_) {
    this->plotTrackEffAfterFit(fitName);
  }

  cout << "=========================================================================" << endl;

  // Print r (z) range in which each barrel layer (endcap wheel) appears. 
  // (Needed by firmware).
  cout<<endl;
  cout<<"--- r range in which stubs in each barrel layer appear ---"<<endl;
  for (const auto& p : mapBarrelLayerMinR_) {
    unsigned int layer = p.first;
    cout<<"   layer = "<<layer
	<<" : "<<mapBarrelLayerMinR_[layer]<<" < r < "<<mapBarrelLayerMaxR_[layer]<<endl;
  }
  cout<<"--- |z| range in which stubs in each endcap wheel appear ---"<<endl;
  for (const auto& p : mapEndcapWheelMinZ_) {
    unsigned int layer = p.first;
    cout<<"   wheel = "<<layer
	<<" : "<<mapEndcapWheelMinZ_[layer]<<" < |z| < "<<mapEndcapWheelMaxZ_[layer]<<endl;
  }

  // Print (r,|z|) range in which each module type (defined in DigitalStub) appears.
  // (Needed by firmware).
  cout<<endl;
  cout<<"--- (r,|z|) range in which each module type (defined in DigitalStub) appears ---"<<endl;
  for (const auto& p : mapModuleTypeMinR_) {
    unsigned int modType = p.first;
    cout<<"   Module type = "<<modType<<setprecision(1)<<
          " : r range = ("<<mapModuleTypeMinR_[modType]<<
                       ","<<mapModuleTypeMaxR_[modType]<<
 	  "); z range = ("<<mapModuleTypeMinZ_[modType]<<
                       ","<<mapModuleTypeMaxZ_[modType]<<")"<<endl;
  }
  // Ugly bodge to allow for modules in barrel layers 1-2 & endcap wheels 3-5 being different.
  cout<<"and in addition"<<endl;
  for (const auto& p : mapExtraAModuleTypeMinR_) {
    unsigned int modType = p.first;
    cout<<"   Module type = "<<modType<<setprecision(1)<<
          " : r range = ("<<mapExtraAModuleTypeMinR_[modType]<<
    	               ","<<mapExtraAModuleTypeMaxR_[modType]<<
          "); z range = ("<<mapExtraAModuleTypeMinZ_[modType]<<","
                          <<mapExtraAModuleTypeMaxZ_[modType]<<")"<<endl;
  }
  cout<<"and in addition"<<endl;
  for (const auto& p : mapExtraBModuleTypeMinR_) {
    unsigned int modType = p.first;
    cout<<"   Module type = "<<modType<<setprecision(1)<<
          " : r range = ("<<mapExtraBModuleTypeMinR_[modType]<<
    	               ","<<mapExtraBModuleTypeMaxR_[modType]<<
          "); z range = ("<<mapExtraBModuleTypeMinZ_[modType]<<","
                          <<mapExtraBModuleTypeMaxZ_[modType]<<")"<<endl;
  }
  cout<<"and in addition"<<endl;
  for (const auto& p : mapExtraCModuleTypeMinR_) {
    unsigned int modType = p.first;
    cout<<"   Module type = "<<modType<<setprecision(1)<<
          " : r range = ("<<mapExtraCModuleTypeMinR_[modType]<<
    	               ","<<mapExtraCModuleTypeMaxR_[modType]<<
          "); z range = ("<<mapExtraCModuleTypeMinZ_[modType]<<","
                          <<mapExtraCModuleTypeMaxZ_[modType]<<")"<<endl;
  }
  cout<<"and in addition"<<endl;
  for (const auto& p : mapExtraDModuleTypeMinR_) {
    unsigned int modType = p.first;
    cout<<"   Module type = "<<modType<<setprecision(1)<<
          " : r range = ("<<mapExtraDModuleTypeMinR_[modType]<<
    	               ","<<mapExtraDModuleTypeMaxR_[modType]<<
          "); z range = ("<<mapExtraDModuleTypeMinZ_[modType]<<","
                          <<mapExtraDModuleTypeMaxZ_[modType]<<")"<<endl;
  }
  cout<<endl;

  if (settings_->hybrid() && not wierdMixedMode) {
    //--- Print summary of tracklet pattern reco
    this->printTrackletSeedFindingPerformance();
    this->printTrackPerformance("TRACKLET");
    this->printHybridDupRemovalPerformance();
  } else {
    //--- Print summary of track-finding performance after HT
    this->printTrackPerformance("HT");
    //--- Optionally print summary of track-finding performance after r-z track filter.
    if (ranRZfilter_) this->printTrackPerformance("RZ");
  }

  //--- Print summary of track-finding performance after helix fit, for each track fitting algorithm used.
  for (const string& fitName : trackFitters_) {
    this->printFitTrackPerformance(fitName);   
  }
  cout << "=========================================================================" << endl;

  if (not settings_->hybrid()) {
    // Check that stub filling was consistent with known limitations of HT firmware design.

    cout<<endl<<"Max. |gradients| of stub lines in HT array is: r-phi = "<<HTrphi::maxLineGrad()<<endl;

    if (HTrphi::maxLineGrad() > 1.) {

      cout<<"WARNING: Line |gradient| exceeds 1, which firmware will not be able to cope with! Please adjust HT array size to avoid this."<<endl;

    } else if (HTrphi::fracErrorsTypeA() > 0.) {

      cout<<"WARNING: Despite line gradients being less than one, some fraction of HT columns have filled cells with no filled neighbours in W, SW or NW direction. Firmware will object to this! ";
      cout<<"This fraction = "<<HTrphi::fracErrorsTypeA()<<" for r-phi HT"<<endl; 

    } else if (HTrphi::fracErrorsTypeB() > 0.) {

      cout<<"WARNING: Despite line gradients being less than one, some fraction of HT columns recorded individual stubs being added to more than two cells! Thomas firmware will object to this! "; 
      cout<<"This fraction = "<<HTrphi::fracErrorsTypeB()<<" for r-phi HT"<<endl;   
    }
  }

  // Check for presence of common MC bug.

  float meanShared = hisFracStubsSharingClus0_->GetMean();
  if (meanShared > 0.01) cout<<endl<<"WARNING: You are using buggy MC. A fraction "<<meanShared<<" of stubs share clusters in the module seed sensor, which front-end electronics forbids."<<endl;

  // Check that the constants in class DegradeBend are up to date.
  float meanFracStubsLost = hisStubKillDegradeBend_->GetMean(2);
  if (meanFracStubsLost > 0.001) cout<<endl<<"WARNING: You should update the constants in class DegradeBend, since some stubs had bend outside the expected window range."<<endl; 

  // Check if GP B approximation cfg params are inconsistent.
  if (bApproxMistake_) cout<<endl<<"WARNING: BApprox cfg params are inconsistent - see printout above."<<endl;
}

//=== Determine "B" parameter, used in GP firmware to allow for tilted modules.

void Histos::trackerGeometryAnalysis( const TrackerGeometryInfo trackerGeometryInfo ) {

  // Don't bother producing summary if user didn't request histograms via TFileService in their cfg.
  if ( ! this->enabled() ) return;

  cout << endl << "=========================================================================" << endl;
  cout         << "--- Fit to cfg params for FPGA-friendly approximation to B parameter in GP & KF ---" << endl;
  cout         << "--- (used to allowed for tilted barrel modules)                                 ---" << endl; 
  // Check that info on the correct number of modules has been stored
  if ( trackerGeometryInfo.moduleZoR().size() != trackerGeometryInfo.barrelNTiltedModules() * trackerGeometryInfo.barrelNLayersWithTiltedModules() * 2 ) {
    cout << "WARNING : Expected " << trackerGeometryInfo.barrelNTiltedModules() * trackerGeometryInfo.barrelNLayersWithTiltedModules() * 2 << " modules, but only recorded info on " << trackerGeometryInfo.moduleZoR().size() << endl;
  }

  TFileDirectory inputDir = fs_->mkdir("InputData");
  graphBVsZoverR_ = inputDir.make<TGraph>( trackerGeometryInfo.moduleZoR().size(), &trackerGeometryInfo.moduleZoR()[0], &trackerGeometryInfo.moduleB()[0] );
  graphBVsZoverR_->SetNameTitle("B vs module Z/R","; Module Z/R; B");
  graphBVsZoverR_->Fit("pol1","q");
  TF1* fittedFunction = graphBVsZoverR_->GetFunction("pol1");
  double gradient     = fittedFunction->GetParameter(1);
  double intercept    = fittedFunction->GetParameter(0);
  cout << "         BApprox_gradient (fitted)  = " << gradient  << endl;
  cout << "         BApprox_intercept (fitted) = " << intercept << endl;
  // Check fitted params consistent with those assumed in cfg file.
  if (settings_->useApproxB()) {
    double gradientDiff  = fabs( gradient  - settings_->bApprox_gradient() );  
    double interceptDiff = fabs( intercept - settings_->bApprox_intercept() ); 
    if ( gradientDiff > 0.001 || interceptDiff > 0.001 ) { // Uncertainty independent of number of events
      cout << endl << "WARNING: fitted parameters inconsistent with those specified in cfg file:" << endl;
      cout << "         BApprox_gradient  (cfg) = " << settings_->bApprox_gradient() << endl;
      cout << "         BApprox_intercept (cfg) = " << settings_->bApprox_intercept() << endl;
      bApproxMistake_ = true; // Note that problem has occurred.
    }
  }

}


}
