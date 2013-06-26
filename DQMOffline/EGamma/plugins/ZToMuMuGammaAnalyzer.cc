#include <iostream>
#include <iomanip>
//

#include "DQMOffline/EGamma/plugins/ZToMuMuGammaAnalyzer.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"


/** \class ZToMuMuGammaAnalyzer
 **
 **
 **  $Id: ZToMuMuGammaAnalyzer
 **  $Date: 2012/06/27 13:04:20 $
 **  authors:
 **   Nancy Marinelli, U. of Notre Dame, US
 **   Nathan Kellams, U. of Notre Dame, US
 **
 ***/

using namespace std;


ZToMuMuGammaAnalyzer::ZToMuMuGammaAnalyzer( const edm::ParameterSet& pset )
{

    fName_                  = pset.getUntrackedParameter<string>("Name");
    verbosity_              = pset.getUntrackedParameter<int>("Verbosity");
    prescaleFactor_         = pset.getUntrackedParameter<int>("prescaleFactor",1);
    standAlone_             = pset.getParameter<bool>("standAlone");
    outputFileName_         = pset.getParameter<string>("OutputFileName");
    isHeavyIon_             = pset.getUntrackedParameter<bool>("isHeavyIon",false);
    triggerEvent_           = pset.getParameter<edm::InputTag>("triggerEvent");
    useTriggerFiltering_    = pset.getParameter<bool>("useTriggerFiltering");
    splitHistosEBEE_        = pset.getParameter<bool>("splitHistosEBEE");
    use2DHistos_            = pset.getParameter<bool>("use2DHistos");
    
    photonProducer_         = pset.getParameter<string>("phoProducer");
    photonCollection_       = pset.getParameter<string>("photonCollection");

    barrelRecHitProducer_   = pset.getParameter<string>("barrelRecHitProducer");
    barrelRecHitCollection_ = pset.getParameter<string>("barrelRecHitCollection");

    endcapRecHitProducer_   = pset.getParameter<string>("endcapRecHitProducer");
    endcapRecHitCollection_ = pset.getParameter<string>("endcapRecHitCollection");

    muonProducer_         = pset.getParameter<string>("muonProducer");
    muonCollection_       = pset.getParameter<string>("muonCollection");
    // Muon selection
    muonMinPt_             = pset.getParameter<double>("muonMinPt");
    minPixStripHits_       = pset.getParameter<int>("minPixStripHits");
    muonMaxChi2_           = pset.getParameter<double>("muonMaxChi2");
    muonMaxDxy_            = pset.getParameter<double>("muonMaxDxy");
    muonMatches_           = pset.getParameter<int>("muonMatches");
    validPixHits_          = pset.getParameter<int>("validPixHits");
    validMuonHits_         = pset.getParameter<int>("validMuonHits");
    muonTrackIso_          = pset.getParameter<double>("muonTrackIso");
    muonTightEta_          = pset.getParameter<double>("muonTightEta");
    // Dimuon selection
    minMumuInvMass_       = pset.getParameter<double>("minMumuInvMass");
    maxMumuInvMass_       = pset.getParameter<double>("maxMumuInvMass");
    // Photon selection
    photonMinEt_             = pset.getParameter<double>("photonMinEt");
    photonMaxEta_            = pset.getParameter<double>("photonMaxEta");
    photonTrackIso_          = pset.getParameter<double>("photonTrackIso");
    // mumuGamma selection
    nearMuonDr_               = pset.getParameter<double>("nearMuonDr");
    nearMuonHcalIso_          = pset.getParameter<double>("nearMuonHcalIso");
    farMuonEcalIso_           = pset.getParameter<double>("farMuonEcalIso");
    farMuonTrackIso_          = pset.getParameter<double>("farMuonTrackIso");
    farMuonMinPt_             = pset.getParameter<double>("farMuonMinPt");
    minMumuGammaInvMass_  = pset.getParameter<double>("minMumuGammaInvMass");
    maxMumuGammaInvMass_  = pset.getParameter<double>("maxMumuGammaInvMass");
    
    parameters_ = pset;

}

ZToMuMuGammaAnalyzer::~ZToMuMuGammaAnalyzer() {}

void ZToMuMuGammaAnalyzer::beginJob()
{
  nEvt_=0;
  nEntry_=0;

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();

  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int    eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int    etBin = parameters_.getParameter<int>("etBin");

  double sumMin = parameters_.getParameter<double>("sumMin");
  double sumMax = parameters_.getParameter<double>("sumMax");
  int    sumBin = parameters_.getParameter<int>("sumBin");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int    etaBin = parameters_.getParameter<int>("etaBin");

  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");

  double r9Min = parameters_.getParameter<double>("r9Min");
  double r9Max = parameters_.getParameter<double>("r9Max");
  int    r9Bin = parameters_.getParameter<int>("r9Bin");

  double hOverEMin = parameters_.getParameter<double>("hOverEMin");
  double hOverEMax = parameters_.getParameter<double>("hOverEMax");
  int    hOverEBin = parameters_.getParameter<int>("hOverEBin");

//   double xMin = parameters_.getParameter<double>("xMin");
//   double xMax = parameters_.getParameter<double>("xMax");
//   int    xBin = parameters_.getParameter<int>("xBin");

//   double yMin = parameters_.getParameter<double>("yMin");
//   double yMax = parameters_.getParameter<double>("yMax");
//   int    yBin = parameters_.getParameter<int>("yBin");

  double numberMin = parameters_.getParameter<double>("numberMin");
  double numberMax = parameters_.getParameter<double>("numberMax");
  int    numberBin = parameters_.getParameter<int>("numberBin");

//   double zMin = parameters_.getParameter<double>("zMin");
//   double zMax = parameters_.getParameter<double>("zMax");
//   int    zBin = parameters_.getParameter<int>("zBin");

//   double rMin = parameters_.getParameter<double>("rMin");
//   double rMax = parameters_.getParameter<double>("rMax");
//   int    rBin = parameters_.getParameter<int>("rBin");

//   double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin");
//   double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax");
//   int    dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin");

//   double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin");
//   double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax");
//   int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");

  double sigmaIetaMin = parameters_.getParameter<double>("sigmaIetaMin");
  double sigmaIetaMax = parameters_.getParameter<double>("sigmaIetaMax");
  int    sigmaIetaBin = parameters_.getParameter<int>("sigmaIetaBin");

//   double eOverPMin = parameters_.getParameter<double>("eOverPMin");
//   double eOverPMax = parameters_.getParameter<double>("eOverPMax");
//   int    eOverPBin = parameters_.getParameter<int>("eOverPBin");

//   double chi2Min = parameters_.getParameter<double>("chi2Min");
//   double chi2Max = parameters_.getParameter<double>("chi2Max");
//   int    chi2Bin = parameters_.getParameter<int>("chi2Bin");


  int reducedEtBin  = etBin/4;
  int reducedEtaBin = etaBin/4;
  int reducedSumBin = sumBin/4;
  int reducedR9Bin  = r9Bin/4;

  ////////////////START OF BOOKING FOR ALL HISTOGRAMS////////////////

  if (dbe_) {


    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/ZToMuMuGamma");
    
    h1_mumuInvMass_      = dbe_->book1D("mumuInvMass","Two muon invariant mass: M (GeV)",etBin,etMin,etMax);
    h1_mumuGammaInvMass_ = dbe_->book1D("mumuGammaInvMass","Two-muon plus gamma invariant mass: M (GeV)",etBin,etMin,etMax);

    ////////////////START OF BOOKING FOR PHOTON-RELATED HISTOGRAMS////////////////

    //// 1D Histograms ////
    
    //ENERGY
    h_phoE_  = dbe_->book1D("phoE","Energy;E (GeV)",eBin,eMin,eMax);
    h_phoEt_ = dbe_->book1D("phoEt","E_{T};E_{T} (GeV)", etBin,etMin,etMax);

    //NUMBER OF PHOTONS
    h_nPho_  = dbe_->book1D("nPho", "Number of Photons per Event;# #gamma", numberBin,numberMin,numberMax);

    //GEOMETRICAL
    h_phoEta_ = dbe_->book1D("phoEta", "#eta;#eta",etaBin,etaMin,etaMax);
    h_phoPhi_ = dbe_->book1D("phoPhi", "#phi;#phi",phiBin,phiMin,phiMax);

    h_scEta_  = dbe_->book1D("scEta", "SuperCluster #eta;#eta",etaBin,etaMin,etaMax);
    h_scPhi_  = dbe_->book1D("scPhi", "SuperCluster #phi;#phi",phiBin,phiMin,phiMax);

    //SHOWER SHAPE
    //r9
    h_r9_      = dbe_->book1D("r9","R9;R9",r9Bin,r9Min, r9Max);
                                                        
    //sigmaIetaIeta
    h_phoSigmaIetaIeta_   = dbe_->book1D("phoSigmaIetaIeta","#sigma_{i#etai#eta};#sigma_{i#etai#eta}",sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);

    //TRACK ISOLATION

    //nTrackIsolSolid
    h_nTrackIsolSolid_       = dbe_->book1D("nIsoTracksSolid","Number Of Tracks in the Solid Iso Cone;# tracks",numberBin,numberMin,numberMax);

    //nTrackIsolHollow
    h_nTrackIsolHollow_      = dbe_->book1D("nIsoTracksHollow","Number Of Tracks in the Hollow Iso Cone;# tracks",numberBin,numberMin,numberMax);
    
    //trackPtSumSolid
    h_trackPtSumSolid_       = dbe_->book1D("isoPtSumSolid","Track P_{T} Sum in the Solid Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);
    
    //trackPtSumHollow
    h_trackPtSumHollow_      = dbe_->book1D("isoPtSumHollow","Track P_{T} Sum in the Hollow Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);

    //CALORIMETER ISOLATION VARIABLES

    //ecal sum
    h_ecalSum_      = dbe_->book1D("ecalSum","Ecal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);

    //hcal sum
    h_hcalSum_      = dbe_->book1D("hcalSum","Hcal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);

    //h over e
    h_hOverE_       = dbe_->book1D("hOverE","H/E;H/E",hOverEBin,hOverEMin,hOverEMax);
    h_h1OverE_      = dbe_->book1D("h1OverE","H/E for Depth 1;H/E",hOverEBin,hOverEMin,hOverEMax);
    h_h2OverE_      = dbe_->book1D("h2OverE","H/E for Depth 2;H/E",hOverEBin,hOverEMin,hOverEMax);


    ///// 2D histograms /////

    if(use2DHistos_){

      //SHOWER SHAPE
      //r9    
      h_r9VsEt_  = dbe_->book2D("r9VsEt2D","R9 vs E_{T};E_{T} (GeV);R9",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r9VsEt_  = dbe_->bookProfile("r9VsEt","Avg R9 vs E_{T};E_{T} (GeV);R9",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r9VsEta_ = dbe_->book2D("r9VsEta2D","R9 vs #eta;#eta;R9",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r9VsEta_ = dbe_->bookProfile("r9VsEta","Avg R9 vs #eta;#eta;R9",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
                                                        
      //sigmaIetaIeta
      h_sigmaIetaIetaVsEta_ = dbe_->book2D("sigmaIetaIetaVsEta2D","#sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",reducedEtaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
      p_sigmaIetaIetaVsEta_ = dbe_->bookProfile("sigmaIetaIetaVsEta","Avg #sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",etaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);

      //e1x5
      h_e1x5VsEt_  = dbe_->book2D("e1x5VsEt2D","E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
      p_e1x5VsEt_  = dbe_->bookProfile("e1x5VsEt","Avg E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
      h_e1x5VsEta_ = dbe_->book2D("e1x5VsEta2D","E1x5 vs #eta;#eta;E1X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
      p_e1x5VsEta_ = dbe_->bookProfile("e1x5VsEta","Avg E1x5 vs #eta;#eta;E1X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);

      //e2x5
      h_e2x5VsEt_  = dbe_->book2D("e2x5VsEt2D","E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
      p_e2x5VsEt_  = dbe_->bookProfile("e2x5VsEt","Avg E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
      h_e2x5VsEta_ = dbe_->book2D("e2x5VsEta2D","E2x5 vs #eta;#eta;E2X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
      p_e2x5VsEta_ = dbe_->bookProfile("e2x5VsEta","Avg E2x5 vs #eta;#eta;E2X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);

      //r1x5
      h_r1x5VsEt_  = dbe_->book2D("r1x5VsEt2D","R1x5 vs E_{T};E_{T} (GeV);R1X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r1x5VsEt_  = dbe_->bookProfile("r1x5VsEt","Avg R1x5 vs E_{T};E_{T} (GeV);R1X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r1x5VsEta_ = dbe_->book2D("r1x5VsEta2D","R1x5 vs #eta;#eta;R1X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r1x5VsEta_ = dbe_->bookProfile("r1x5VsEta","Avg R1x5 vs #eta;#eta;R1X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);

      //r2x5
      h_r2x5VsEt_  = dbe_->book2D("r2x5VsEt2D","R2x5 vs E_{T};E_{T} (GeV);R2X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r2x5VsEt_  = dbe_->bookProfile("r2x5VsEt","Avg R2x5 vs E_{T};E_{T} (GeV);R2X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r2x5VsEta_ = dbe_->book2D("r2x5VsEta2D","R2x5 vs #eta;#eta;R2X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r2x5VsEta_ = dbe_->bookProfile("r2x5VsEta","Avg R2x5 vs #eta;#eta;R2X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);

      //maxEXtalOver3x3
      h_maxEXtalOver3x3VsEt_  = dbe_->book2D("maxEXtalOver3x3VsEt2D","(Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",reducedEtBin,etMin,etMax,r9Bin,r9Min,r9Max);
      p_maxEXtalOver3x3VsEt_  = dbe_->bookProfile("maxEXtalOver3x3VsEt","Avg (Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_maxEXtalOver3x3VsEta_ = dbe_->book2D("maxEXtalOver3x3VsEta2D","(Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",reducedEtaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      p_maxEXtalOver3x3VsEta_ = dbe_->bookProfile("maxEXtalOver3x3VsEta","Avg (Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);

      //TRACK ISOLATION

      //nTrackIsolSolid
      h_nTrackIsolSolidVsEt_   = dbe_->book2D("nIsoTracksSolidVsEt2D","Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
      p_nTrackIsolSolidVsEt_   = dbe_->bookProfile("nIsoTracksSolidVsEt","Avg Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
      h_nTrackIsolSolidVsEta_  = dbe_->book2D("nIsoTracksSolidVsEta2D","Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      p_nTrackIsolSolidVsEta_  = dbe_->bookProfile("nIsoTracksSolidVsEta","Avg Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);

      //nTrackIsolHollow
      h_nTrackIsolHollowVsEt_  = dbe_->book2D("nIsoTracksHollowVsEt2D","Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
      p_nTrackIsolHollowVsEt_  = dbe_->bookProfile("nIsoTracksHollowVsEt","Avg Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
      h_nTrackIsolHollowVsEta_ = dbe_->book2D("nIsoTracksHollowVsEta2D","Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      p_nTrackIsolHollowVsEta_ = dbe_->bookProfile("nIsoTracksHollowVsEta","Avg Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
    
      //trackPtSumSolid
      h_trackPtSumSolidVsEt_   = dbe_->book2D("isoPtSumSolidVsEt2D","Track P_{T} Sum in the Solid Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumSolidVsEt_   = dbe_->bookProfile("isoPtSumSolidVsEt","Avg Track P_{T} Sum in the Solid Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
      h_trackPtSumSolidVsEta_  = dbe_->book2D("isoPtSumSolidVsEta2D","Track P_{T} Sum in the Solid Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumSolidVsEta_  = dbe_->bookProfile("isoPtSumSolidVsEta","Avg Track P_{T} Sum in the Solid Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
    
      //trackPtSumHollow
      h_trackPtSumHollowVsEt_  = dbe_->book2D("isoPtSumHollowVsEt2D","Track P_{T} Sum in the Hollow Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumHollowVsEt_  = dbe_->bookProfile("isoPtSumHollowVsEt","Avg Track P_{T} Sum in the Hollow Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
      h_trackPtSumHollowVsEta_ = dbe_->book2D("isoPtSumHollowVsEta2D","Track P_{T} Sum in the Hollow Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumHollowVsEta_ = dbe_->bookProfile("isoPtSumHollowVsEta","Avg Track P_{T} Sum in the Hollow Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);

      //CALORIMETER ISOLATION VARIABLES

      //ecal sum
      h_ecalSumVsEt_  = dbe_->book2D("ecalSumVsEt2D","Ecal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_ecalSumVsEt_  = dbe_->bookProfile("ecalSumVsEt","Avg Ecal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
      h_ecalSumVsEta_ = dbe_->book2D("ecalSumVsEta2D","Ecal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_ecalSumVsEta_ = dbe_->bookProfile("ecalSumVsEta","Avg Ecal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);

      //hcal sum
      h_hcalSumVsEt_  = dbe_->book2D("hcalSumVsEt2D","Hcal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_hcalSumVsEt_  = dbe_->bookProfile("hcalSumVsEt","Avg Hcal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
      h_hcalSumVsEta_ = dbe_->book2D("hcalSumVsEta2D","Hcal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_hcalSumVsEta_ = dbe_->bookProfile("hcalSumVsEta","Avg Hcal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);

      //h over e
      p_hOverEVsEt_   = dbe_->bookProfile("hOverEVsEt","Avg H/E vs Et;E_{T} (GeV);H/E",etBin,etMin,etMax,hOverEBin,hOverEMin,hOverEMax);
      p_hOverEVsEta_  = dbe_->bookProfile("hOverEVsEta","Avg H/E vs #eta;#eta;H/E",etaBin,etaMin,etaMax,hOverEBin,hOverEMin,hOverEMax);

    }

      ////////////  Barrel only histos ////////////

    if(splitHistosEBEE_){  

      //EB ENERGY
      h_phoEBarrel_  = dbe_->book1D("phoEBarrel","Energy for Barrel;E (GeV)",eBin,eMin,eMax);
      h_phoEtBarrel_ = dbe_->book1D("phoEtBarrel","E_{T};E_{T} (GeV)", etBin,etMin,etMax);
    
      //EB NUMBER OF PHOTONS
      h_nPhoBarrel_  = dbe_->book1D("nPhoBarrel","Number of Photons per Event;# #gamma", numberBin,numberMin,numberMax);
      
      //EB GEOMETRICAL
      h_phoEtaBarrel_ = dbe_->book1D("phoEtaBarrel","#eta;#eta",etaBin,etaMin,etaMax);
      h_phoPhiBarrel_ = dbe_->book1D("phoPhiBarrel","#phi;#phi",phiBin,phiMin,phiMax);
    
      h_scEtaBarrel_  = dbe_->book1D("scEtaBarrel","SuperCluster #eta;#eta",etaBin,etaMin,etaMax);
      h_scPhiBarrel_  = dbe_->book1D("scPhiBarrel","SuperCluster #phi;#phi",phiBin,phiMin,phiMax);
    
      //EB SHOWER SHAPE
      //EB r9
      h_r9Barrel_      = dbe_->book1D("r9Barrel","R9;R9",r9Bin,r9Min, r9Max);
      h_r9VsEtBarrel_  = dbe_->book2D("r9VsEt2DBarrel","R9 vs E_{T};E_{T} (GeV);R9",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r9VsEtBarrel_  = dbe_->bookProfile("r9VsEtBarrel","Avg R9 vs E_{T};E_{T} (GeV);R9",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r9VsEtaBarrel_ = dbe_->book2D("r9VsEta2DBarrel","R9 vs #eta;#eta;R9",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r9VsEtaBarrel_ = dbe_->bookProfile("r9VsEtaBarrel","Avg R9 vs #eta;#eta;R9",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EB sigmaIetaIeta
      h_phoSigmaIetaIetaBarrel_   = dbe_->book1D("phoSigmaIetaIetaBarrel","#sigma_{i#etai#eta};#sigma_{i#etai#eta}",sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
      h_sigmaIetaIetaVsEtaBarrel_ = dbe_->book2D("sigmaIetaIetaVsEta2DBarrel","#sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",reducedEtaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
      p_sigmaIetaIetaVsEtaBarrel_ = dbe_->bookProfile("sigmaIetaIetaVsEtaBarrel","Avg #sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",etaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
      
      //EB e1x5
      h_e1x5VsEtBarrel_  = dbe_->book2D("e1x5VsEt2DBarrel","E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
      p_e1x5VsEtBarrel_  = dbe_->bookProfile("e1x5VsEtBarrel","Avg E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
      h_e1x5VsEtaBarrel_ = dbe_->book2D("e1x5VsEta2DBarrel","E1x5 vs #eta;#eta;E1X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
      p_e1x5VsEtaBarrel_ = dbe_->bookProfile("e1x5VsEtaBarrel","Avg E1x5 vs #eta;#eta;E1X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);
      
      //EB e2x5
      h_e2x5VsEtBarrel_  = dbe_->book2D("e2x5VsEt2DBarrel","E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
      p_e2x5VsEtBarrel_  = dbe_->bookProfile("e2x5VsEtBarrel","Avg E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
      h_e2x5VsEtaBarrel_ = dbe_->book2D("e2x5VsEta2DBarrel","E2x5 vs #eta;#eta;E2X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
      p_e2x5VsEtaBarrel_ = dbe_->bookProfile("e2x5VsEtaBarrel","Avg E2x5 vs #eta;#eta;E2X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);
      
      //EB r1x5
      h_r1x5VsEtBarrel_  = dbe_->book2D("r1x5VsEt2DBarrel","R1x5 vs E_{T};E_{T} (GeV);R1X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r1x5VsEtBarrel_  = dbe_->bookProfile("r1x5VsEtBarrel","Avg R1x5 vs E_{T};E_{T} (GeV);R1X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r1x5VsEtaBarrel_ = dbe_->book2D("r1x5VsEta2DBarrel","R1x5 vs #eta;#eta;R1X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r1x5VsEtaBarrel_ = dbe_->bookProfile("r1x5VsEtaBarrel","Avg R1x5 vs #eta;#eta;R1X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EB r2x5
      h_r2x5VsEtBarrel_  = dbe_->book2D("r2x5VsEt2DBarrel","R2x5 vs E_{T};E_{T} (GeV);R2X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r2x5VsEtBarrel_  = dbe_->bookProfile("r2x5VsEtBarrel","Avg R2x5 vs E_{T};E_{T} (GeV);R2X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r2x5VsEtaBarrel_ = dbe_->book2D("r2x5VsEta2DBarrel","R2x5 vs #eta;#eta;R2X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r2x5VsEtaBarrel_ = dbe_->bookProfile("r2x5VsEtaBarrel","Avg R2x5 vs #eta;#eta;R2X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EB maxEXtalOver3x3
      h_maxEXtalOver3x3VsEtBarrel_  = dbe_->book2D("maxEXtalOver3x3VsEt2DBarrel","(Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",reducedEtBin,etMin,etMax,r9Bin,r9Min,r9Max);
      p_maxEXtalOver3x3VsEtBarrel_  = dbe_->bookProfile("maxEXtalOver3x3VsEtBarrel","Avg (Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_maxEXtalOver3x3VsEtaBarrel_ = dbe_->book2D("maxEXtalOver3x3VsEta2DBarrel","(Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",reducedEtaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      p_maxEXtalOver3x3VsEtaBarrel_ = dbe_->bookProfile("maxEXtalOver3x3VsEtaBarrel","Avg (Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EB TRACK ISOLATION
      
      //EB nTrackIsolSolid
      h_nTrackIsolSolidBarrel_       = dbe_->book1D("nIsoTracksSolidBarrel","Number Of Tracks in the Solid Iso Cone;# tracks",numberBin,numberMin,numberMax);
      h_nTrackIsolSolidVsEtBarrel_   = dbe_->book2D("nIsoTracksSolidVsEt2DBarrel","Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
      p_nTrackIsolSolidVsEtBarrel_   = dbe_->bookProfile("nIsoTracksSolidVsEtBarrel","Avg Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
      h_nTrackIsolSolidVsEtaBarrel_  = dbe_->book2D("nIsoTracksSolidVsEta2DBarrel","Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      p_nTrackIsolSolidVsEtaBarrel_  = dbe_->bookProfile("nIsoTracksSolidVsEtaBarrel","Avg Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      
      //EB nTrackIsolHollow
      h_nTrackIsolHollowBarrel_      = dbe_->book1D("nIsoTracksHollowBarrel","Number Of Tracks in the Hollow Iso Cone;# tracks",numberBin,numberMin,numberMax);
      h_nTrackIsolHollowVsEtBarrel_  = dbe_->book2D("nIsoTracksHollowVsEt2DBarrel","Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
      p_nTrackIsolHollowVsEtBarrel_  = dbe_->bookProfile("nIsoTracksHollowVsEtBarrel","Avg Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
      h_nTrackIsolHollowVsEtaBarrel_ = dbe_->book2D("nIsoTracksHollowVsEta2DBarrel","Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      p_nTrackIsolHollowVsEtaBarrel_ = dbe_->bookProfile("nIsoTracksHollowVsEtaBarrel","Avg Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      
      //EB trackPtSumSolid
      h_trackPtSumSolidBarrel_       = dbe_->book1D("isoPtSumSolidBarrel","Track P_{T} Sum in the Solid Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);
      h_trackPtSumSolidVsEtBarrel_   = dbe_->book2D("isoPtSumSolidVsEt2DBarrel","Track P_{T} Sum in the Solid Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumSolidVsEtBarrel_   = dbe_->bookProfile("isoPtSumSolidVsEtBarrel","Avg Track P_{T} Sum in the Solid Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
      h_trackPtSumSolidVsEtaBarrel_  = dbe_->book2D("isoPtSumSolidVsEta2DBarrel","Track P_{T} Sum in the Solid Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumSolidVsEtaBarrel_  = dbe_->bookProfile("isoPtSumSolidVsEtaBarrel","Avg Track P_{T} Sum in the Solid Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EB trackPtSumHollow
      h_trackPtSumHollowBarrel_      = dbe_->book1D("isoPtSumHollowBarrel","Track P_{T} Sum in the Hollow Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);
      h_trackPtSumHollowVsEtBarrel_  = dbe_->book2D("isoPtSumHollowVsEt2DBarrel","Track P_{T} Sum in the Hollow Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumHollowVsEtBarrel_  = dbe_->bookProfile("isoPtSumHollowVsEtBarrel","Avg Track P_{T} Sum in the Hollow Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
      h_trackPtSumHollowVsEtaBarrel_ = dbe_->book2D("isoPtSumHollowVsEta2DBarrel","Track P_{T} Sum in the Hollow Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumHollowVsEtaBarrel_ = dbe_->bookProfile("isoPtSumHollowVsEtaBarrel","Avg Track P_{T} Sum in the Hollow Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EB CALORIMETER ISOLATION VARIABLES
      
      //EB ecal sum
      h_ecalSumBarrel_      = dbe_->book1D("ecalSumBarrel","Ecal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);
      h_ecalSumVsEtBarrel_  = dbe_->book2D("ecalSumVsEt2DBarrel","Ecal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_ecalSumVsEtBarrel_  = dbe_->bookProfile("ecalSumVsEtBarrel","Avg Ecal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
      h_ecalSumVsEtaBarrel_ = dbe_->book2D("ecalSumVsEta2DBarrel","Ecal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_ecalSumVsEtaBarrel_ = dbe_->bookProfile("ecalSumVsEtaBarrel","Avg Ecal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EB hcal sum
      h_hcalSumBarrel_      = dbe_->book1D("hcalSumBarrel","Hcal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);
      h_hcalSumVsEtBarrel_  = dbe_->book2D("hcalSumVsEt2DBarrel","Hcal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_hcalSumVsEtBarrel_  = dbe_->bookProfile("hcalSumVsEtBarrel","Avg Hcal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
      h_hcalSumVsEtaBarrel_ = dbe_->book2D("hcalSumVsEta2DBarrel","Hcal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_hcalSumVsEtaBarrel_ = dbe_->bookProfile("hcalSumVsEtaBarrel","Avg Hcal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EB h over e
      h_hOverEBarrel_       = dbe_->book1D("hOverEBarrel","H/E;H/E",hOverEBin,hOverEMin,hOverEMax);
      p_hOverEVsEtBarrel_   = dbe_->bookProfile("hOverEVsEtBarrel","Avg H/E vs Et;E_{T} (GeV);H/E",etBin,etMin,etMax,hOverEBin,hOverEMin,hOverEMax);
      p_hOverEVsEtaBarrel_  = dbe_->bookProfile("hOverEVsEtaBarrel","Avg H/E vs #eta;#eta;H/E",etaBin,etaMin,etaMax,hOverEBin,hOverEMin,hOverEMax);
      h_h1OverEBarrel_      = dbe_->book1D("h1OverEBarrel","H/E for Depth 1;H/E",hOverEBin,hOverEMin,hOverEMax);
      h_h2OverEBarrel_      = dbe_->book1D("h2OverEBarrel","H/E for Depth 2;H/E",hOverEBin,hOverEMin,hOverEMax);

    
      ////////////  Endcap only histos ////////////
      
      //EE ENERGY
      h_phoEtEndcap_ = dbe_->book1D("phoEtEndcap","E_{T};E_{T} (GeV)", etBin,etMin,etMax);
      h_phoEEndcap_  = dbe_->book1D("phoEEndcap","Energy for Endcap;E (GeV)",eBin,eMin,eMax);
      
      //EE NUMBER OF PHOTONS
      h_nPhoEndcap_  = dbe_->book1D("nPhoEndcap","Number of Photons per Event;# #gamma", numberBin,numberMin,numberMax);
      
      //EE GEOMETRICAL
      h_phoEtaEndcap_ = dbe_->book1D("phoEtaEndcap","#eta;#eta",etaBin,etaMin,etaMax);
      h_phoPhiEndcap_ = dbe_->book1D("phoPhiEndcap","#phi;#phi",phiBin,phiMin,phiMax);
      
      h_scEtaEndcap_  = dbe_->book1D("scEtaEndcap","SuperCluster #eta;#eta",etaBin,etaMin,etaMax);
      h_scPhiEndcap_  = dbe_->book1D("scPhiEndcap","SuperCluster #phi;#phi",phiBin,phiMin,phiMax);
      
      //EE SHOWER SHAPE
      //EE r9
      h_r9Endcap_      = dbe_->book1D("r9Endcap","R9;R9",r9Bin,r9Min, r9Max);
      h_r9VsEtEndcap_  = dbe_->book2D("r9VsEt2DEndcap","R9 vs E_{T};E_{T} (GeV);R9",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r9VsEtEndcap_  = dbe_->bookProfile("r9VsEtEndcap","Avg R9 vs E_{T};E_{T} (GeV);R9",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r9VsEtaEndcap_ = dbe_->book2D("r9VsEta2DEndcap","R9 vs #eta;#eta;R9",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r9VsEtaEndcap_ = dbe_->bookProfile("r9VsEtaEndcap","Avg R9 vs #eta;#eta;R9",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EE sigmaIetaIeta
      h_phoSigmaIetaIetaEndcap_   = dbe_->book1D("phoSigmaIetaIetaEndcap","#sigma_{i#etai#eta};#sigma_{i#etai#eta}",sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
      h_sigmaIetaIetaVsEtaEndcap_ = dbe_->book2D("sigmaIetaIetaVsEta2DEndcap","#sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",reducedEtaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
      p_sigmaIetaIetaVsEtaEndcap_ = dbe_->bookProfile("sigmaIetaIetaVsEtaEndcap","Avg #sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",etaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
      
      //EE e1x5
      h_e1x5VsEtEndcap_  = dbe_->book2D("e1x5VsEt2DEndcap","E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
      p_e1x5VsEtEndcap_  = dbe_->bookProfile("e1x5VsEtEndcap","Avg E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
      h_e1x5VsEtaEndcap_ = dbe_->book2D("e1x5VsEta2DEndcap","E1x5 vs #eta;#eta;E1X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
      p_e1x5VsEtaEndcap_ = dbe_->bookProfile("e1x5VsEtaEndcap","Avg E1x5 vs #eta;#eta;E1X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);
      
      //EE e2x5
      h_e2x5VsEtEndcap_  = dbe_->book2D("e2x5VsEt2DEndcap","E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
      p_e2x5VsEtEndcap_  = dbe_->bookProfile("e2x5VsEtEndcap","Avg E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
      h_e2x5VsEtaEndcap_ = dbe_->book2D("e2x5VsEta2DEndcap","E2x5 vs #eta;#eta;E2X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
      p_e2x5VsEtaEndcap_ = dbe_->bookProfile("e2x5VsEtaEndcap","Avg E2x5 vs #eta;#eta;E2X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);
      
      //EE r1x5
      h_r1x5VsEtEndcap_  = dbe_->book2D("r1x5VsEt2DEndcap","R1x5 vs E_{T};E_{T} (GeV);R1X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r1x5VsEtEndcap_  = dbe_->bookProfile("r1x5VsEtEndcap","Avg R1x5 vs E_{T};E_{T} (GeV);R1X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r1x5VsEtaEndcap_ = dbe_->book2D("r1x5VsEta2DEndcap","R1x5 vs #eta;#eta;R1X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r1x5VsEtaEndcap_ = dbe_->bookProfile("r1x5VsEtaEndcap","Avg R1x5 vs #eta;#eta;R1X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EE r2x5
      h_r2x5VsEtEndcap_  = dbe_->book2D("r2x5VsEt2DEndcap","R2x5 vs E_{T};E_{T} (GeV);R2X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
      p_r2x5VsEtEndcap_  = dbe_->bookProfile("r2x5VsEtEndcap","Avg R2x5 vs E_{T};E_{T} (GeV);R2X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_r2x5VsEtaEndcap_ = dbe_->book2D("r2x5VsEta2DEndcap","R2x5 vs #eta;#eta;R2X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
      p_r2x5VsEtaEndcap_ = dbe_->bookProfile("r2x5VsEtaEndcap","Avg R2x5 vs #eta;#eta;R2X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EE maxEXtalOver3x3
      h_maxEXtalOver3x3VsEtEndcap_  = dbe_->book2D("maxEXtalOver3x3VsEt2DEndcap","(Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",reducedEtBin,etMin,etMax,r9Bin,r9Min,r9Max);
      p_maxEXtalOver3x3VsEtEndcap_  = dbe_->bookProfile("maxEXtalOver3x3VsEtEndcap","Avg (Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
      h_maxEXtalOver3x3VsEtaEndcap_ = dbe_->book2D("maxEXtalOver3x3VsEta2DEndcap","(Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",reducedEtaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      p_maxEXtalOver3x3VsEtaEndcap_ = dbe_->bookProfile("maxEXtalOver3x3VsEtaEndcap","Avg (Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
      
      //EE TRACK ISOLATION
      
      //EE nTrackIsolSolid
      h_nTrackIsolSolidEndcap_       = dbe_->book1D("nIsoTracksSolidEndcap","Number Of Tracks in the Solid Iso Cone;# tracks",numberBin,numberMin,numberMax);
      h_nTrackIsolSolidVsEtEndcap_   = dbe_->book2D("nIsoTracksSolidVsEt2DEndcap","Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
      p_nTrackIsolSolidVsEtEndcap_   = dbe_->bookProfile("nIsoTracksSolidVsEtEndcap","Avg Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
      h_nTrackIsolSolidVsEtaEndcap_  = dbe_->book2D("nIsoTracksSolidVsEta2DEndcap","Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      p_nTrackIsolSolidVsEtaEndcap_  = dbe_->bookProfile("nIsoTracksSolidVsEtaEndcap","Avg Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      
      //EE nTrackIsolHollow
      h_nTrackIsolHollowEndcap_      = dbe_->book1D("nIsoTracksHollowEndcap","Number Of Tracks in the Hollow Iso Cone;# tracks",numberBin,numberMin,numberMax);
      h_nTrackIsolHollowVsEtEndcap_  = dbe_->book2D("nIsoTracksHollowVsEt2DEndcap","Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
      p_nTrackIsolHollowVsEtEndcap_  = dbe_->bookProfile("nIsoTracksHollowVsEtEndcap","Avg Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
      h_nTrackIsolHollowVsEtaEndcap_ = dbe_->book2D("nIsoTracksHollowVsEta2DEndcap","Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      p_nTrackIsolHollowVsEtaEndcap_ = dbe_->bookProfile("nIsoTracksHollowVsEtaEndcap","Avg Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
      
      //EE trackPtSumSolid
      h_trackPtSumSolidEndcap_       = dbe_->book1D("isoPtSumSolidEndcap","Track P_{T} Sum in the Solid Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);
      h_trackPtSumSolidVsEtEndcap_   = dbe_->book2D("isoPtSumSolidVsEt2DEndcap","Track P_{T} Sum in the Solid Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumSolidVsEtEndcap_   = dbe_->bookProfile("isoPtSumSolidVsEtEndcap","Avg Track P_{T} Sum in the Solid Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
      h_trackPtSumSolidVsEtaEndcap_  = dbe_->book2D("isoPtSumSolidVsEta2DEndcap","Track P_{T} Sum in the Solid Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumSolidVsEtaEndcap_  = dbe_->bookProfile("isoPtSumSolidVsEtaEndcap","Avg Track P_{T} Sum in the Solid Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EE trackPtSumHollow
      h_trackPtSumHollowEndcap_      = dbe_->book1D("isoPtSumHollowEndcap","Track P_{T} Sum in the Hollow Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);
      h_trackPtSumHollowVsEtEndcap_  = dbe_->book2D("isoPtSumHollowVsEt2DEndcap","Track P_{T} Sum in the Hollow Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumHollowVsEtEndcap_  = dbe_->bookProfile("isoPtSumHollowVsEtEndcap","Avg Track P_{T} Sum in the Hollow Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
      h_trackPtSumHollowVsEtaEndcap_ = dbe_->book2D("isoPtSumHollowVsEta2DEndcap","Track P_{T} Sum in the Hollow Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_trackPtSumHollowVsEtaEndcap_ = dbe_->bookProfile("isoPtSumHollowVsEtaEndcap","Avg Track P_{T} Sum in the Hollow Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EE CALORIMETER ISOLATION VARIABLES
      
      //EE ecal sum
      h_ecalSumEndcap_      = dbe_->book1D("ecalSumEndcap","Ecal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);
      h_ecalSumVsEtEndcap_  = dbe_->book2D("ecalSumVsEt2DEndcap","Ecal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_ecalSumVsEtEndcap_  = dbe_->bookProfile("ecalSumVsEtEndcap","Avg Ecal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
      h_ecalSumVsEtaEndcap_ = dbe_->book2D("ecalSumVsEta2DEndcap","Ecal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_ecalSumVsEtaEndcap_ = dbe_->bookProfile("ecalSumVsEtaEndcap","Avg Ecal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EE hcal sum
      h_hcalSumEndcap_      = dbe_->book1D("hcalSumEndcap","Hcal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);
      h_hcalSumVsEtEndcap_  = dbe_->book2D("hcalSumVsEt2DEndcap","Hcal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
      p_hcalSumVsEtEndcap_  = dbe_->bookProfile("hcalSumVsEtEndcap","Avg Hcal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
      h_hcalSumVsEtaEndcap_ = dbe_->book2D("hcalSumVsEta2DEndcap","Hcal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
      p_hcalSumVsEtaEndcap_ = dbe_->bookProfile("hcalSumVsEtaEndcap","Avg Hcal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);
      
      //EE h over e
      h_hOverEEndcap_       = dbe_->book1D("hOverEEndcap","H/E;H/E",hOverEBin,hOverEMin,hOverEMax);
      p_hOverEVsEtEndcap_   = dbe_->bookProfile("hOverEVsEtEndcap","Avg H/E vs Et;E_{T} (GeV);H/E",etBin,etMin,etMax,hOverEBin,hOverEMin,hOverEMax);
      p_hOverEVsEtaEndcap_  = dbe_->bookProfile("hOverEVsEtaEndcap","Avg H/E vs #eta;#eta;H/E",etaBin,etaMin,etaMax,hOverEBin,hOverEMin,hOverEMax);
      h_h1OverEEndcap_      = dbe_->book1D("h1OverEEndcap","H/E for Depth 1;H/E",hOverEBin,hOverEMin,hOverEMax);
      h_h2OverEEndcap_      = dbe_->book1D("h2OverEEndcap","H/E for Depth 2;H/E",hOverEBin,hOverEMin,hOverEMax);
      
    }//end if(splitHistosEBEE)
    
  }//end if(dbe_)

}//end BeginJob

void ZToMuMuGammaAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  using namespace edm;

  if (nEvt_% prescaleFactor_ ) return;
  nEvt_++;
  LogInfo("ZToMuMuGammaAnalyzer") << "ZToMuMuGammaAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";

  // Get the trigger results
  bool validTriggerEvent=true;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  trigger::TriggerEvent triggerEvent;
  e.getByLabel(triggerEvent_,triggerEventHandle);
  if(!triggerEventHandle.isValid()) {
    edm::LogInfo("PhotonAnalyzer") << "Error! Can't get the product "<< triggerEvent_.label() << endl;
    validTriggerEvent=false;
  }
  if(validTriggerEvent) triggerEvent = *(triggerEventHandle.product());

  // Get the reconstructed photons
  bool validPhotons=true;
  Handle<reco::PhotonCollection> photonHandle;
  reco::PhotonCollection photonCollection;
  e.getByLabel(photonProducer_, photonCollection_ , photonHandle);
  if ( !photonHandle.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< photonCollection_ << endl;
    validPhotons=false;
  }
  if(validPhotons) photonCollection = *(photonHandle.product());

  // Get the PhotonId objects
  bool validloosePhotonID=true;
  Handle<edm::ValueMap<bool> > loosePhotonFlag;
  edm::ValueMap<bool> loosePhotonID;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDLoose", loosePhotonFlag);
  if ( !loosePhotonFlag.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< "PhotonCutBasedIDLoose" << endl;
    validloosePhotonID=false;
  }
  if (validloosePhotonID) loosePhotonID = *(loosePhotonFlag.product());

  bool validtightPhotonID=true;
  Handle<edm::ValueMap<bool> > tightPhotonFlag;
  edm::ValueMap<bool> tightPhotonID;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDTight", tightPhotonFlag);
  if ( !tightPhotonFlag.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< "PhotonCutBasedIDTight" << endl;
    validtightPhotonID=false;
  }
  if (validtightPhotonID) tightPhotonID = *(tightPhotonFlag.product());

  // Get the reconstructed muons
  bool validMuons=true;
  Handle<reco::MuonCollection> muonHandle;
  reco::MuonCollection muonCollection;
  e.getByLabel(muonProducer_, muonCollection_ , muonHandle);
  if ( !muonHandle.isValid()) {
    edm::LogInfo("ZToMuMuGammaAnalyzer") << "Error! Can't get the product "<< muonCollection_ << endl;
    validMuons=false;
  }
  if(validMuons) muonCollection = *(muonHandle.product());

  // Get the beam spot
  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByLabel("offlineBeamSpot", bsHandle);
  if (!bsHandle.isValid()) {
      edm::LogError("TrackerOnlyConversionProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
      return;
  }
  const reco::BeamSpot &thebs = *bsHandle.product();

  //Prepare list of photon-related HLT filter names
  vector<int> Keys;
  for(uint filterIndex=0;filterIndex<triggerEvent.sizeFilters();++filterIndex){  //loop over all trigger filters in event (i.e. filters passed)
    string label = triggerEvent.filterTag(filterIndex).label();
    if(label.find( "Photon" ) != string::npos ) {  //get photon-related filters
      for(uint filterKeyIndex=0;filterKeyIndex<triggerEvent.filterKeys(filterIndex).size();++filterKeyIndex){  //loop over keys to objects passing this filter
        Keys.push_back(triggerEvent.filterKeys(filterIndex)[filterKeyIndex]);  //add keys to a vector for later reference
      }
    }
  }
  
  // sort Keys vector in ascending order
  // and erases duplicate entries from the vector
  sort(Keys.begin(),Keys.end());
  for ( uint i=0 ; i<Keys.size() ; )
    {
      if (i!=(Keys.size()-1))
        {
          if (Keys[i]==Keys[i+1]) Keys.erase(Keys.begin()+i+1) ;
          else ++i ;
        }
      else ++i ;
    }

  //photon counters
  int nPho = 0;
  int nPhoBarrel = 0;
  int nPhoEndcap = 0;

  ////////////// event selection
  if ( muonCollection.size() < 2 ) return;

  for( reco::MuonCollection::const_iterator  iMu = muonCollection.begin(); iMu != muonCollection.end(); iMu++) {
    if ( !basicMuonSelection (*iMu) ) continue;
 
    for( reco::MuonCollection::const_iterator  iMu2 = iMu+1; iMu2 != muonCollection.end(); iMu2++) {
      if ( !basicMuonSelection (*iMu2) ) continue;
      if ( iMu->charge()*iMu2->charge() > 0) continue;

      if ( !muonSelection(*iMu,thebs) && !muonSelection(*iMu2,thebs) ) continue;
    
      float mumuMass = mumuInvMass(*iMu,*iMu2) ;
      if ( mumuMass <  minMumuInvMass_  ||  mumuMass >  maxMumuInvMass_ ) continue;

      h1_mumuInvMass_ -> Fill (mumuMass);      

      if (  photonCollection.size() < 1 ) continue;

      reco::Muon nearMuon;
      reco::Muon farMuon;
      for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
        if ( !photonSelection (*iPho) ) continue;

        DeltaR<reco::Muon, reco::Photon> deltaR;
        double dr1 = deltaR(*iMu, *iPho);
        double dr2 = deltaR(*iMu2,*iPho);
        double drNear = dr1;
        if (dr1 < dr2) {
          nearMuon =*iMu ; farMuon  = *iMu2; drNear = dr1;
        } else {
          nearMuon = *iMu2; farMuon  = *iMu; drNear = dr2;
        }
        
        if ( nearMuon.isolationR03().hadEt > nearMuonHcalIso_ )  continue;
        if ( farMuon.isolationR03().sumPt > farMuonTrackIso_ )  continue;
        if ( farMuon.isolationR03().emEt  > farMuonEcalIso_ )  continue;
        if ( farMuon.pt() < farMuonMinPt_ )       continue;
        if ( drNear > nearMuonDr_)                continue;
        
        float mumuGammaMass = mumuGammaInvMass(*iMu,*iMu2,*iPho) ;
        if ( mumuGammaMass < minMumuGammaInvMass_ || mumuGammaMass > maxMumuGammaInvMass_ ) continue;

        //counter: number of photons
        nPho++;
        
        //PHOTON RELATED HISTOGRAMS
        h1_mumuGammaInvMass_ ->Fill (mumuGammaMass);
        
        //ENERGY        
        h_phoE_  ->Fill ((*iPho).energy());
        h_phoEt_ ->Fill ((*iPho).et());
        
        //GEOMETRICAL
        h_phoEta_ ->Fill ((*iPho).eta());
        h_phoPhi_ ->Fill ((*iPho).phi());
        
        h_scEta_  ->Fill ((*iPho).superCluster()->eta());
        h_scPhi_  ->Fill ((*iPho).superCluster()->phi());

        //SHOWER SHAPE
        h_r9_     ->Fill ((*iPho).r9());

        h_phoSigmaIetaIeta_    ->Fill((*iPho).sigmaIetaIeta());

        //TRACK ISOLATION
        
        h_nTrackIsolSolid_      ->Fill((*iPho).nTrkSolidConeDR04());
       
        h_nTrackIsolHollow_      ->Fill((*iPho).nTrkHollowConeDR04());    
        
        h_trackPtSumSolid_       ->Fill((*iPho).trkSumPtSolidConeDR04());        
 
        h_trackPtSumHollow_      ->Fill((*iPho).trkSumPtSolidConeDR04());

        //CALORIMETER ISOLATION
        
        h_ecalSum_      ->Fill((*iPho).ecalRecHitSumEtConeDR04());
        
        h_hcalSum_      ->Fill((*iPho).hcalTowerSumEtConeDR04());
       
        h_hOverE_       ->Fill((*iPho).hadTowOverEm());
        h_h1OverE_      ->Fill((*iPho).hadTowDepth1OverEm());
        h_h2OverE_      ->Fill((*iPho).hadTowDepth2OverEm());

        
        //// 2D Histos ////

        if(use2DHistos_){

          //SHOWER SHAPE
          h_r9VsEt_ ->Fill ((*iPho).et(),(*iPho).r9());
          p_r9VsEt_ ->Fill ((*iPho).et(),(*iPho).r9());
          h_r9VsEta_->Fill ((*iPho).eta(),(*iPho).r9());
          p_r9VsEta_->Fill ((*iPho).eta(),(*iPho).r9());
          
          h_e1x5VsEta_->Fill((*iPho).eta(),(*iPho).e1x5());
          p_e1x5VsEta_->Fill((*iPho).eta(),(*iPho).e1x5());
          h_e1x5VsEt_ ->Fill((*iPho).et(), (*iPho).e1x5());
          p_e1x5VsEt_ ->Fill((*iPho).et(), (*iPho).e1x5());
          
          h_e2x5VsEta_->Fill((*iPho).eta(),(*iPho).e2x5());
          p_e2x5VsEta_->Fill((*iPho).eta(),(*iPho).e2x5());
          h_e2x5VsEt_ ->Fill((*iPho).et(), (*iPho).e2x5());
          p_e2x5VsEt_ ->Fill((*iPho).et(), (*iPho).e2x5());
          
          h_r1x5VsEta_->Fill((*iPho).eta(),(*iPho).r1x5());
          p_r1x5VsEta_->Fill((*iPho).eta(),(*iPho).r1x5());
          h_r1x5VsEt_ ->Fill((*iPho).et(), (*iPho).r1x5());
          p_r1x5VsEt_ ->Fill((*iPho).et(), (*iPho).r1x5());
          
          h_r2x5VsEta_->Fill((*iPho).eta(),(*iPho).r2x5());
          p_r2x5VsEta_->Fill((*iPho).eta(),(*iPho).r2x5());
          h_r2x5VsEt_ ->Fill((*iPho).et(), (*iPho).r2x5());
          p_r2x5VsEt_ ->Fill((*iPho).et(), (*iPho).r2x5());
          
          h_maxEXtalOver3x3VsEta_->Fill((*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3());
          p_maxEXtalOver3x3VsEta_->Fill((*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3());
          h_maxEXtalOver3x3VsEt_ ->Fill((*iPho).et(), (*iPho).maxEnergyXtal()/(*iPho).e3x3());
          p_maxEXtalOver3x3VsEt_ ->Fill((*iPho).et(), (*iPho).maxEnergyXtal()/(*iPho).e3x3());
          
          h_sigmaIetaIetaVsEta_  ->Fill((*iPho).eta(),(*iPho).sigmaIetaIeta());
          p_sigmaIetaIetaVsEta_  ->Fill((*iPho).eta(),(*iPho).sigmaIetaIeta());
          
          //TRACK ISOLATION
          h_nTrackIsolSolidVsEt_  ->Fill((*iPho).et(), (*iPho).nTrkSolidConeDR04());
          p_nTrackIsolSolidVsEt_  ->Fill((*iPho).et(), (*iPho).nTrkSolidConeDR04());
          h_nTrackIsolSolidVsEta_ ->Fill((*iPho).eta(),(*iPho).nTrkSolidConeDR04());
          p_nTrackIsolSolidVsEta_ ->Fill((*iPho).eta(),(*iPho).nTrkSolidConeDR04());
          
          h_nTrackIsolHollowVsEt_  ->Fill((*iPho).et(), (*iPho).nTrkHollowConeDR04());
          p_nTrackIsolHollowVsEt_  ->Fill((*iPho).et(), (*iPho).nTrkHollowConeDR04());
          h_nTrackIsolHollowVsEta_ ->Fill((*iPho).eta(),(*iPho).nTrkHollowConeDR04());
          p_nTrackIsolHollowVsEta_ ->Fill((*iPho).eta(),(*iPho).nTrkHollowConeDR04());
          
          h_trackPtSumSolidVsEt_   ->Fill((*iPho).et(), (*iPho).trkSumPtSolidConeDR04());
          p_trackPtSumSolidVsEt_   ->Fill((*iPho).et(), (*iPho).trkSumPtSolidConeDR04());
          h_trackPtSumSolidVsEta_  ->Fill((*iPho).eta(),(*iPho).trkSumPtSolidConeDR04());
          p_trackPtSumSolidVsEta_  ->Fill((*iPho).eta(),(*iPho).trkSumPtSolidConeDR04());
          
          h_trackPtSumHollowVsEt_  ->Fill((*iPho).et(), (*iPho).trkSumPtHollowConeDR04());
          p_trackPtSumHollowVsEt_  ->Fill((*iPho).et(), (*iPho).trkSumPtHollowConeDR04());
          h_trackPtSumHollowVsEta_ ->Fill((*iPho).eta(),(*iPho).trkSumPtHollowConeDR04());
          p_trackPtSumHollowVsEta_ ->Fill((*iPho).eta(),(*iPho).trkSumPtHollowConeDR04());
          
          //CALORIMETER ISOLATION
          h_ecalSumVsEt_  ->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
          p_ecalSumVsEt_  ->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
          h_ecalSumVsEta_ ->Fill((*iPho).eta(),(*iPho).ecalRecHitSumEtConeDR04());
          p_ecalSumVsEta_ ->Fill((*iPho).eta(),(*iPho).ecalRecHitSumEtConeDR04());
          
          h_hcalSumVsEt_  ->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
          p_hcalSumVsEt_  ->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
          h_hcalSumVsEta_ ->Fill((*iPho).eta(),(*iPho).hcalTowerSumEtConeDR04());
          p_hcalSumVsEta_ ->Fill((*iPho).eta(),(*iPho).hcalTowerSumEtConeDR04());
          
          p_hOverEVsEt_   ->Fill((*iPho).et(), (*iPho).hadTowOverEm());    
          p_hOverEVsEta_  ->Fill((*iPho).eta(),(*iPho).hadTowOverEm());
          
        }

        
        ///////////// BARREL ONLY /////////////////

        if(iPho->isEB() && splitHistosEBEE_){
          //EB photon counter
          nPhoBarrel++;

          //EB ENERGY        
          h_phoEBarrel_  ->Fill ((*iPho).energy());
          h_phoEtBarrel_ ->Fill ((*iPho).et());

          //EB GEOMETRICAL
          h_phoEtaBarrel_ ->Fill ((*iPho).eta());
          h_phoPhiBarrel_ ->Fill ((*iPho).phi());
          
          h_scEtaBarrel_  ->Fill ((*iPho).superCluster()->eta());
          h_scPhiBarrel_  ->Fill ((*iPho).superCluster()->phi());
          
          //EB SHOWER SHAPE
          h_r9Barrel_     ->Fill ((*iPho).r9());
          h_r9VsEtBarrel_ ->Fill ((*iPho).et(),(*iPho).r9());
          p_r9VsEtBarrel_ ->Fill ((*iPho).et(),(*iPho).r9());
          h_r9VsEtaBarrel_ ->Fill ((*iPho).eta(),(*iPho).r9());
          p_r9VsEtaBarrel_ ->Fill ((*iPho).eta(),(*iPho).r9());
          
          h_e1x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).e1x5());
          p_e1x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).e1x5());
          h_e1x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).e1x5());
          p_e1x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).e1x5());
        
          h_e2x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).e2x5());
          p_e2x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).e2x5());
          h_e2x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).e2x5());
          p_e2x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).e2x5());
          
          h_r1x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).r1x5());
          p_r1x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).r1x5());
          h_r1x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).r1x5());
          p_r1x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).r1x5());
          
          h_r2x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).r2x5());
          p_r2x5VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).r2x5());
          h_r2x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).r2x5());
          p_r2x5VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).r2x5());
          
          h_maxEXtalOver3x3VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3());
          p_maxEXtalOver3x3VsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3());
          h_maxEXtalOver3x3VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).maxEnergyXtal()/(*iPho).e3x3());
          p_maxEXtalOver3x3VsEtBarrel_ ->Fill((*iPho).et(), (*iPho).maxEnergyXtal()/(*iPho).e3x3());

          h_phoSigmaIetaIetaBarrel_    ->Fill((*iPho).sigmaIetaIeta());
          h_sigmaIetaIetaVsEtaBarrel_  ->Fill((*iPho).eta(),(*iPho).sigmaIetaIeta());
          p_sigmaIetaIetaVsEtaBarrel_  ->Fill((*iPho).eta(),(*iPho).sigmaIetaIeta());
          
          //EB TRACK ISOLATION
          
          h_nTrackIsolSolidBarrel_      ->Fill((*iPho).nTrkSolidConeDR04());
          h_nTrackIsolSolidVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).nTrkSolidConeDR04());
          p_nTrackIsolSolidVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).nTrkSolidConeDR04());
          h_nTrackIsolSolidVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).nTrkSolidConeDR04());
          p_nTrackIsolSolidVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).nTrkSolidConeDR04());
          
          h_nTrackIsolHollowBarrel_      ->Fill((*iPho).nTrkHollowConeDR04());    
          h_nTrackIsolHollowVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).nTrkHollowConeDR04());
          p_nTrackIsolHollowVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).nTrkHollowConeDR04());
          h_nTrackIsolHollowVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).nTrkHollowConeDR04());
          p_nTrackIsolHollowVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).nTrkHollowConeDR04());
        
          h_trackPtSumSolidBarrel_       ->Fill((*iPho).trkSumPtSolidConeDR04());        
          h_trackPtSumSolidVsEtBarrel_   ->Fill((*iPho).et(), (*iPho).trkSumPtSolidConeDR04());
          p_trackPtSumSolidVsEtBarrel_   ->Fill((*iPho).et(), (*iPho).trkSumPtSolidConeDR04());
          h_trackPtSumSolidVsEtaBarrel_  ->Fill((*iPho).eta(),(*iPho).trkSumPtSolidConeDR04());
          p_trackPtSumSolidVsEtaBarrel_  ->Fill((*iPho).eta(),(*iPho).trkSumPtSolidConeDR04());
          
          h_trackPtSumHollowBarrel_      ->Fill((*iPho).trkSumPtSolidConeDR04());
          h_trackPtSumHollowVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).trkSumPtHollowConeDR04());
          p_trackPtSumHollowVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).trkSumPtHollowConeDR04());
          h_trackPtSumHollowVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).trkSumPtHollowConeDR04());
          p_trackPtSumHollowVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).trkSumPtHollowConeDR04());
          
          //EB CALORIMETER ISOLATION
          
          h_ecalSumBarrel_      ->Fill((*iPho).ecalRecHitSumEtConeDR04());
          h_ecalSumVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
          p_ecalSumVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
          h_ecalSumVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).ecalRecHitSumEtConeDR04());
          p_ecalSumVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).ecalRecHitSumEtConeDR04());
        
          h_hcalSumBarrel_      ->Fill((*iPho).hcalTowerSumEtConeDR04());
          h_hcalSumVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
          p_hcalSumVsEtBarrel_  ->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
          h_hcalSumVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).hcalTowerSumEtConeDR04());
          p_hcalSumVsEtaBarrel_ ->Fill((*iPho).eta(),(*iPho).hcalTowerSumEtConeDR04());
          
          h_hOverEBarrel_       ->Fill((*iPho).hadTowOverEm());
          p_hOverEVsEtBarrel_   ->Fill((*iPho).et(), (*iPho).hadTowOverEm());    
          p_hOverEVsEtaBarrel_  ->Fill((*iPho).eta(),(*iPho).hadTowOverEm());
          h_h1OverEBarrel_      ->Fill((*iPho).hadTowDepth1OverEm());
          h_h2OverEBarrel_      ->Fill((*iPho).hadTowDepth2OverEm());
          
        }

        ///////////// ENDCAP ONLY /////////////////

        if(iPho->isEE() && splitHistosEBEE_){
          //EE photon counter
          nPhoEndcap++;

          //EE ENERGY        
          h_phoEEndcap_  ->Fill ((*iPho).energy());
          h_phoEtEndcap_ ->Fill ((*iPho).et());

          //EE GEOMETRICAL
          h_phoEtaEndcap_ ->Fill ((*iPho).eta());
          h_phoPhiEndcap_ ->Fill ((*iPho).phi());
          
          h_scEtaEndcap_  ->Fill ((*iPho).superCluster()->eta());
          h_scPhiEndcap_  ->Fill ((*iPho).superCluster()->phi());
          
          //EE SHOWER SHAPE
          h_r9Endcap_     ->Fill ((*iPho).r9());
          h_r9VsEtEndcap_ ->Fill ((*iPho).et(),(*iPho).r9());
          p_r9VsEtEndcap_ ->Fill ((*iPho).et(),(*iPho).r9());
          h_r9VsEtaEndcap_ ->Fill ((*iPho).eta(),(*iPho).r9());
          p_r9VsEtaEndcap_ ->Fill ((*iPho).eta(),(*iPho).r9());
          
          h_e1x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).e1x5());
          p_e1x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).e1x5());
          h_e1x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).e1x5());
          p_e1x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).e1x5());
        
          h_e2x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).e2x5());
          p_e2x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).e2x5());
          h_e2x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).e2x5());
          p_e2x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).e2x5());
          
          h_r1x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).r1x5());
          p_r1x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).r1x5());
          h_r1x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).r1x5());
          p_r1x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).r1x5());
          
          h_r2x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).r2x5());
          p_r2x5VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).r2x5());
          h_r2x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).r2x5());
          p_r2x5VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).r2x5());
          
          h_maxEXtalOver3x3VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3());
          p_maxEXtalOver3x3VsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3());
          h_maxEXtalOver3x3VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).maxEnergyXtal()/(*iPho).e3x3());
          p_maxEXtalOver3x3VsEtEndcap_ ->Fill((*iPho).et(), (*iPho).maxEnergyXtal()/(*iPho).e3x3());

          h_phoSigmaIetaIetaEndcap_    ->Fill((*iPho).sigmaIetaIeta());
          h_sigmaIetaIetaVsEtaEndcap_  ->Fill((*iPho).eta(),(*iPho).sigmaIetaIeta());
          p_sigmaIetaIetaVsEtaEndcap_  ->Fill((*iPho).eta(),(*iPho).sigmaIetaIeta());
          
          //EE TRACK ISOLATION
          
          h_nTrackIsolSolidEndcap_      ->Fill((*iPho).nTrkSolidConeDR04());
          h_nTrackIsolSolidVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).nTrkSolidConeDR04());
          p_nTrackIsolSolidVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).nTrkSolidConeDR04());
          h_nTrackIsolSolidVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).nTrkSolidConeDR04());
          p_nTrackIsolSolidVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).nTrkSolidConeDR04());
          
          h_nTrackIsolHollowEndcap_      ->Fill((*iPho).nTrkHollowConeDR04());    
          h_nTrackIsolHollowVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).nTrkHollowConeDR04());
          p_nTrackIsolHollowVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).nTrkHollowConeDR04());
          h_nTrackIsolHollowVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).nTrkHollowConeDR04());
          p_nTrackIsolHollowVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).nTrkHollowConeDR04());
        
          h_trackPtSumSolidEndcap_       ->Fill((*iPho).trkSumPtSolidConeDR04());        
          h_trackPtSumSolidVsEtEndcap_   ->Fill((*iPho).et(), (*iPho).trkSumPtSolidConeDR04());
          p_trackPtSumSolidVsEtEndcap_   ->Fill((*iPho).et(), (*iPho).trkSumPtSolidConeDR04());
          h_trackPtSumSolidVsEtaEndcap_  ->Fill((*iPho).eta(),(*iPho).trkSumPtSolidConeDR04());
          p_trackPtSumSolidVsEtaEndcap_  ->Fill((*iPho).eta(),(*iPho).trkSumPtSolidConeDR04());
          
          h_trackPtSumHollowEndcap_      ->Fill((*iPho).trkSumPtSolidConeDR04());
          h_trackPtSumHollowVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).trkSumPtHollowConeDR04());
          p_trackPtSumHollowVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).trkSumPtHollowConeDR04());
          h_trackPtSumHollowVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).trkSumPtHollowConeDR04());
          p_trackPtSumHollowVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).trkSumPtHollowConeDR04());
          
          //EE CALORIMETER ISOLATION
          
          h_ecalSumEndcap_      ->Fill((*iPho).ecalRecHitSumEtConeDR04());
          h_ecalSumVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
          p_ecalSumVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
          h_ecalSumVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).ecalRecHitSumEtConeDR04());
          p_ecalSumVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).ecalRecHitSumEtConeDR04());
        
          h_hcalSumEndcap_      ->Fill((*iPho).hcalTowerSumEtConeDR04());
          h_hcalSumVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
          p_hcalSumVsEtEndcap_  ->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
          h_hcalSumVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).hcalTowerSumEtConeDR04());
          p_hcalSumVsEtaEndcap_ ->Fill((*iPho).eta(),(*iPho).hcalTowerSumEtConeDR04());
          
          h_hOverEEndcap_       ->Fill((*iPho).hadTowOverEm());
          p_hOverEVsEtEndcap_   ->Fill((*iPho).et(), (*iPho).hadTowOverEm());    
          p_hOverEVsEtaEndcap_  ->Fill((*iPho).eta(),(*iPho).hadTowOverEm());
          h_h1OverEEndcap_      ->Fill((*iPho).hadTowDepth1OverEm());
          h_h2OverEEndcap_      ->Fill((*iPho).hadTowDepth2OverEm());
          
        }

        
      } //end photon loop

      h_nPho_ ->Fill (float(nPho));

      if(splitHistosEBEE_){
        h_nPhoBarrel_ ->Fill (float(nPhoBarrel));
        h_nPhoEndcap_ ->Fill (float(nPhoEndcap));
      }
    } //end inner muon loop

  } //end outer muon loop

}//End of Analyze method

void ZToMuMuGammaAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& setup)
{
  if(!standAlone_){dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/ZToMuMuGamma");}
}

void ZToMuMuGammaAnalyzer::endJob()
{
  //dbe_->showDirStructure();
  if(standAlone_){
    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/ZToMuMuGamma");
    dbe_->save(outputFileName_);
  }
}

bool ZToMuMuGammaAnalyzer::basicMuonSelection ( const reco::Muon & mu) {
  bool result=true;
  if (!mu.innerTrack().isNonnull())    result=false;
  if (!mu.globalTrack().isNonnull())   result=false;
  if ( !mu.isGlobalMuon() )            result=false; 
  if ( mu.pt() < muonMinPt_ )                  result=false;
  if ( fabs(mu.eta())>2.4 )            result=false;

  int pixHits=0;
  int tkHits=0;
  if ( mu.innerTrack().isNonnull() ) {
    pixHits=mu.innerTrack()->hitPattern().numberOfValidPixelHits();
    tkHits=mu.innerTrack()->hitPattern().numberOfValidStripHits();
  }

  if ( pixHits+tkHits < minPixStripHits_ ) result=false;
  
  return result;  
}

bool ZToMuMuGammaAnalyzer::muonSelection ( const reco::Muon & mu,  const reco::BeamSpot& beamSpot) {
  bool result=true;
  if ( mu.globalTrack()->normalizedChi2() > muonMaxChi2_ )          result=false;
  if ( fabs( mu.globalTrack()->dxy(beamSpot)) > muonMaxDxy_ )       result=false;
  if ( mu.numberOfMatches() < muonMatches_ )                                   result=false;

  if ( mu.track()-> hitPattern().numberOfValidPixelHits() <  validPixHits_ )     result=false;
  if ( mu.globalTrack()->hitPattern().numberOfValidMuonHits() < validMuonHits_ ) result=false;
  if ( !mu.isTrackerMuon() )                                        result=false;
  // track isolation 
  if ( mu.isolationR03().sumPt > muonTrackIso_ )                                result=false;
  if ( fabs(mu.eta())>  muonTightEta_ )                                         result=false;
 
  return result;  
}

bool ZToMuMuGammaAnalyzer::photonSelection ( const reco::Photon & pho) {
  bool result=true;
  if ( pho.pt() < photonMinEt_ )          result=false;
  if ( fabs(pho.eta())> photonMaxEta_ )   result=false;
  if ( pho.isEBEEGap() )       result=false;
  //  if ( pho.trkSumPtHollowConeDR04() >   photonTrackIso_ )   result=false; // check how to exclude the muon track (which muon track).

  return result;  
}

float ZToMuMuGammaAnalyzer::mumuInvMass(const reco::Muon & mu1,const reco::Muon & mu2 )
 {
  math::XYZTLorentzVector p12 = mu1.p4()+mu2.p4() ;
  float mumuMass2 = p12.Dot(p12) ;
  float invMass = sqrt(mumuMass2) ;
  return invMass ;
 }

float ZToMuMuGammaAnalyzer::mumuGammaInvMass(const reco::Muon & mu1,const reco::Muon & mu2, const reco::Photon& pho )
 {
   math::XYZTLorentzVector p12 = mu1.p4()+mu2.p4()+pho.p4() ;
   float Mass2 = p12.Dot(p12) ;
   float invMass = sqrt(Mass2) ;
   return invMass ;
 }

