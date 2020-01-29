#ifndef __HISTOS_H__
#define __HISTOS_H__

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerGeometryInfo.h"

#include "boost/numeric/ublas/matrix.hpp"
using  boost::numeric::ublas::matrix;

#include <vector>
#include <map>
#include <string>

using namespace std;

class TH1F;
class TH2F;
class TH2Poly;
class TF1;
class TProfile;
class TGraphAsymmErrors;
class TGraph;
class TEfficiency;


namespace TMTT {

class InputData;
class TP;
class Sector;
class HTrphi;
class Get3Dtracks;
class L1fittedTrack;
class L1fittedTrk4and5;

class Histos {

public:
  // Store cfg parameters.
  Histos(const Settings* settings);

  virtual ~Histos(){}

  // Book & fill all histograms.
  virtual void book();
  virtual void fill(const InputData& inputData, const matrix<Sector>& mSectors, const matrix<HTrphi>& mHtPhis, 
    	    const matrix<Get3Dtracks> mGet3Dtrks, const std::map<std::string,std::vector<L1fittedTrack>>& fittedTracks);

  // Print tracking performance summary & make tracking efficiency histograms.
  virtual void endJobAnalysis();

  // Determine "B" parameter, used in GP firmware to allow for tilted modules.
  virtual void trackerGeometryAnalysis( const TrackerGeometryInfo trackerGeometryInfo );

  // Did user request output histograms via the TFileService in their cfg?
  virtual  bool available() const {return fs_.isAvailable();}

  // Should histograms be produced?
  virtual bool enabled() const {return ( settings_->enableHistos() && available() );}

protected:

  // Book histograms for specific topics.
  virtual TFileDirectory bookInputData();
  virtual TFileDirectory bookEtaPhiSectors();
  virtual TFileDirectory bookRphiHT();
  virtual TFileDirectory bookRZfilters();
  virtual TFileDirectory bookStudyBusyEvents();
  virtual TFileDirectory bookTrackCands(string tName);
  virtual map<string, TFileDirectory> bookTrackFitting();

  // Fill histograms for specific topics.
  virtual void fillInputData(const InputData& inputData);
  virtual void fillEtaPhiSectors(const InputData& inputData, const matrix<Sector>& mSectors);
  virtual void fillRphiHT(const matrix<HTrphi>& mHtRphis);
  virtual void fillRZfilters(const matrix<Get3Dtracks>& mGet3Dtrks);
  virtual void fillStudyBusyEvents(const InputData& inputData, const matrix<Sector>& mSectors, const matrix<HTrphi>& mHtRphis, 
    		           const matrix<Get3Dtracks>& mGet3Dtrks);
  virtual void fillTrackCands(const InputData& inputData, const vector<L1track3D>& tracks, string tName);
  virtual void fillTrackFitting(const InputData& inputData, const std::map<std::string,std::vector<L1fittedTrack>>& fittedTracks);

  // Produce plots of tracking efficiency after HZ or after r-z track filter (run at end of job)
  virtual TFileDirectory plotTrackEfficiency(string tName);
  // Produce plots of tracking efficiency after track fit (run at end of job).
  virtual TFileDirectory plotTrackEffAfterFit(string fitName);

  // For Hybrid tracking
  // Produce plots of tracklet seed finding efficiency before track reco
  virtual void           plotTrackletSeedEfficiency() {};
  // Produce plots of hybrid duplicate removal efficiency after track reco
  virtual void           plotHybridDupRemovalEfficiency() {};

  virtual void makeEfficiencyPlot( TFileDirectory &inputDir, TEfficiency* outputEfficiency, TH1F* pass, TH1F* all, TString name, TString title );

  // Print summary of track-finding performance after track pattern reco.
  virtual void printTrackPerformance(string tName);

  // Print summary of track-finding performance after helix fit for given track fitter.
  virtual void printFitTrackPerformance(string fitName);

  // For Hybrid tracking
  // Print summary of seed finding and extrapolation performance during track pattern reco.
  virtual void printTrackletSeedFindingPerformance() {};

  // Print summary of duplicate removal performance after track pattern reco.
  virtual void printHybridDupRemovalPerformance() {};

  // Understand why not all tracking particles were reconstructed.
  // Returns list of tracking particles that were not reconstructed and an integer indicating why.
  // Only considers TP used for algorithmic efficiency measurement.
  virtual map<const TP*, string> diagnoseTracking(const vector<TP>& allTPs, const vector<L1track3D>& tracks,
					  bool withRZfilter) const;

 protected:

  // Configuration parameters.
  const Settings *settings_; 
  unsigned int genMinStubLayers_;
  unsigned int numPhiSectors_;
  unsigned int numEtaRegions_;
  float houghMinPt_;
  unsigned int houghNbinsPt_;
  unsigned int houghNbinsPhi_;
  float chosenRofZ_;
  vector<string> trackFitters_;
  vector<string> useRZfilter_;
  bool ranRZfilter_;
  bool resPlotOpt_;

  edm::Service<TFileService> fs_;

  // Histograms of input data.
  TH1F*     hisNumEvents_;
  TProfile* profNumStubs_;
  TH1F* hisStubsVsEta_;
  TH1F* hisStubsVsR_;
  TH2F* hisStubsVsRVsZ_;
  TH2F* hisStubsModuleVsRVsZ_;
  TH2F* hisStubsModuleTiltVsZ_;
  TH2F* hisStubsdPhiCorrectionVsZ_;
  TH2F* hisStubsVsRVsPhi_;
  TH2F* hisStubsModuleVsRVsPhi_;

  TH2F* hisStubsVsRVsZ_outerModuleAtSmallerR_;
  TH2F* hisStubsVsRVsPhi_outerModuleAtSmallerR_;

  TProfile* profNumTPs_;
  TH1F* hisNumStubsPerTP_;
  TH1F* hisNumPSStubsPerTP_;
  TH1F* hisNum2SStubsPerTP_;
  TH1F* hisNumLayersPerTP_;
  TH1F* hisNumPSLayersPerTP_;
  TH1F* hisNum2SLayersPerTP_;

  TH1F* hisNumLayersPerTP_lowPt_;
  TH1F* hisNumPSLayersPerTP_lowPt_;
  TH1F* hisNum2SLayersPerTP_lowPt_;

  TH1F* hisNumLayersPerTP_mediumPt_;
  TH1F* hisNumPSLayersPerTP_mediumPt_;
  TH1F* hisNum2SLayersPerTP_mediumPt_;

  TH1F* hisNumLayersPerTP_highPt_;
  TH1F* hisNumPSLayersPerTP_highPt_;
  TH1F* hisNum2SLayersPerTP_highPt_;

  TH1F* hisNumLayersPerTP_muons_;
  TH1F* hisNumPSLayersPerTP_muons_;
  TH1F* hisNum2SLayersPerTP_muons_;

  TH1F* hisNumLayersPerTP_electrons_;
  TH1F* hisNumPSLayersPerTP_electrons_;
  TH1F* hisNum2SLayersPerTP_electrons_;

  TH1F* hisNumLayersPerTP_pions_;
  TH1F* hisNumPSLayersPerTP_pions_;
  TH1F* hisNum2SLayersPerTP_pions_;

  TProfile* hisStubKillFE_;
  TProfile* hisStubIneffiVsInvPt_;
  TProfile* hisStubIneffiVsEta_;
  TProfile* hisStubKillDegradeBend_;
  TH1F* hisPtStub_;
  TH1F* hisPtResStub_;
  TH1F* hisBendFilterPower_;
  TH1F* hisDelPhiStub_;
  TH1F* hisDelPhiResStub_;
  TH1F* hisDelPhiResStub_tilted_;
  TH1F* hisDelPhiResStub_notTilted_;
  TH1F* hisBendStub_;
  TH1F* hisBendResStub_;
  TH1F* hisNumMergedBend_;
  TH2F* hisBendVsLayerOrRingPS_;
  TH2F* hisBendVsLayerOrRing2S_;
  TH2F* hisBendFEVsLayerOrRingPS_;
  TH2F* hisBendFEVsLayerOrRing2S_;
  TH1F* hisPhiStubVsPhiTP_;
  TH1F* hisPhiStubVsPhi0TP_;
  TH1F* hisPhi0StubVsPhi0TP_;
  TH1F* hisPhi0StubVsPhi0TPres_;
  TH1F* hisPhiStubVsPhi65TP_;
  TH1F* hisPhi65StubVsPhi65TP_;
  TH1F* hisPhi65StubVsPhi65TPres_;
  TH1F* hisPitchOverSep_;
  TH1F* hisRhoParameter_;
  TH2F* hisAlphaCheck_;
  TH1F* hisFracStubsSharingClus0_;
  TH1F* hisFracStubsSharingClus1_;

  // Histograms of B
  TH1F* hisStubB_;
  TH1F* hisStubBApproxDiff_tilted_;
  TGraph* graphBVsZoverR_;

  // Histograms checking that (eta,phi) sector definition is good.
  TH1F* hisFracStubsInSec_;
  TH1F* hisFracStubsInEtaSec_;
  TH1F* hisFracStubsInPhiSec_;
  TH1F* hisNumSecsPerStub_;
  TH1F* hisNumEtaSecsPerStub_;
  TH1F* hisNumPhiSecsPerStub_;
  TH1F* hisNumStubsPerSec_;
  TProfile* profNumStubsPerEtaSec_;
  TH2F* hisLayerIDvsEtaSec_;
  TH2F* hisLayerIDreducedvsEtaSec_;

  // Histograms checking filling of r-phi HT array.
  TH2Poly* hisArrayHT_;
  TF1* hisStubHT_;
  TH1F* hisIncStubsPerHT_;
  TH1F* hisExcStubsPerHT_;
  TH2F* hisNumStubsInCellVsEta_;
  TH1F* hisStubsOnRphiTracksPerHT_;
  TH1F* hisHTstubsPerTrack_;
  TH1F* hisHTmBin_;
  TH1F* hisHTcBin_;

  // Histograms about r-z track filters (or other filters applied after r-phi HT array).
  TH1F* hisNumZtrkSeedCombinations_;
  TH1F* hisNumSeedCombinations_;
  TH1F* hisNumGoodSeedCombinations_;
  TH1F* hisCorrelationZTrk_;

  // Histograms for studying freak, large events with too many stubs.
  TH1F*     hisNumBusySecsInPerEvent_;
  TH1F*     hisNumBusySecsOutPerEvent_;
  TProfile* profFracBusyInVsEtaReg_;
  TProfile* profFracBusyOutVsEtaReg_;
  TProfile* profFracStubsKilledVsEtaReg_;
  TProfile* profFracTracksKilledVsEtaReg_;
  TProfile* profFracTracksKilledVsInvPt_;
  TProfile* profFracTPKilledVsEta_;
  TProfile* profFracTPKilledVsInvPt_;
  TH1F*     hisNumTPkilledBusySec_;
  map<string, TH1F*> hisNumInputStubs_;
  map<string, TH1F*> hisQoverPtInputStubs_;
  map<string, TH1F*> hisNumOutputStubs_;
  map<string, TH1F*> hisNumTracks_; 
  map<string, TH1F*> hisNumStubsPerTrack_; 
  map<string, TH1F*> hisTrackQoverPt_; 
  map<string, TH1F*> hisTrackPurity_; 
  map<string, TH1F*> hisNumTPphysics_; 
  map<string, TH1F*> hisNumTPpileup_; 
  map<string, TH1F*> hisSumPtTPphysics_; 
  map<string, TH1F*> hisSumPtTPpileup_; 

  // Histograms studying 3D track candidates found by Hough Transform or r-z Track Filter.
  map<string, TProfile*> profNumTrackCands_;
  map<string, TProfile*> profNumTracksVsEta_;
  map<string, TH1F*>     hisNumTracksVsQoverPt_;
  map<string, TH1F*>     hisNumTrksPerSect_;
  map<string, TH1F*>     hisNumTrksPerNon_;
  map<string, TProfile*> profStubsOnTracks_;
  map<string, TProfile*> profStubsOnTracksVsEta_;
  map<string, TH1F*>     hisStubsOnTracksPerSect_;
  map<string, TH1F*>     hisStubsOnTracksPerNon_;
  map<string, TH1F*>     hisUniqueStubsOnTrksPerSect_;
  map<string, TH1F*>     hisUniqueStubsOnTrksPerNon_;
  map<string, TH1F*>     hisStubsPerTrack_;
  map<string, TH1F*>     hisLayersPerTrack_;
  map<string, TH1F*>     hisPSLayersPerTrack_;
  map<string, TH1F*>     hisLayersPerTrueTrack_;
  map<string, TH1F*>     hisPSLayersPerTrueTrack_;

  map<string, TH1F*>     hisNumStubsPerLink_;
  map<string, TH2F*>     hisNumStubsVsLink_;
  map<string, TProfile*> profMeanStubsPerLink_;
  map<string, TH1F*>     hisNumTrksPerLink_;
  map<string, TH2F*>     hisNumTrksVsLink_;
  map<string, TProfile*> profMeanTrksPerLink_;

  map<string, TProfile*> profExcessStubsPerTrackVsPt_;
  map<string, TH1F*>     hisFracMatchStubsOnTracks_;
  map<string, TH1F*> hisDeltaBendTrue_;
  map<string, TH1F*> hisDeltaBendFake_;
  map<string, TProfile*> profFracTrueStubsVsLayer_;
  map<string, TProfile*> profDupTracksVsEta_;
  map<string, TProfile*> profDupTracksVsInvPt_;
  //map<string, TH2F*> hisWrongSignStubRZ_pBend_;
  //map<string, TH2F*> hisWrongSignStubRZ_nBend_;

  // Histos of track params after HT.
  map<string, TH1F*> hisQoverPt_;
  map<string, TH1F*> hisPhi0_;
  map<string, TH1F*> hisEta_;
  map<string, TH1F*> hisZ0_;

  // Histograms of track parameter resolution after HT transform.
  map<string, TH1F*> hisQoverPtRes_;
  map<string, TH1F*> hisPhi0Res_;
  map<string, TH1F*> hisEtaRes_;
  map<string, TH1F*> hisZ0Res_;

  map<string, TH2F*> hisRecoVsTrueQinvPt_;
  map<string, TH2F*> hisRecoVsTruePhi0_;
  map<string, TH2F*> hisRecoVsTrueD0_;
  map<string, TH2F*> hisRecoVsTrueZ0_;
  map<string, TH2F*> hisRecoVsTrueEta_;

  // Diagnosis of failed tracking.
  map<string, TH1F*> hisRecoFailureReason_;
  map<string, TH1F*> hisRecoFailureLayer_;

  map<string, TH1F*> hisNumStubsOnLayer_;

  // Histos used for denominator of tracking efficiency plots.
  TH1F* hisTPinvptForEff_;
  TH1F* hisTPptForEff_;
  TH1F* hisTPetaForEff_;
  TH1F* hisTPphiForEff_;
  TH1F* hisTPd0ForEff_;
  TH1F* hisTPz0ForEff_;
  //
  TH1F* hisTPinvptForAlgEff_;
  TH1F* hisTPptForAlgEff_;
  TH1F* hisTPetaForAlgEff_;
  TH1F* hisTPphiForAlgEff_;
  TH1F* hisTPd0ForAlgEff_;
  TH1F* hisTPz0ForAlgEff_;
  //
  TH1F* hisTPphisecForAlgEff_;
  TH1F* hisTPetasecForAlgEff_;
  TH1F* hisTPinvptForAlgEff_inJetPtG30_;
  TH1F* hisTPinvptForAlgEff_inJetPtG100_;
  TH1F* hisTPinvptForAlgEff_inJetPtG200_;

  // Histograms used to make efficiency plots with 3D track candidates prior to fit.
  map<string, TH1F*> hisRecoTPinvptForEff_;
  map<string, TH1F*> hisRecoTPptForEff_;
  map<string, TH1F*> hisRecoTPetaForEff_;
  map<string, TH1F*> hisRecoTPphiForEff_;
  //
  map<string, TH1F*> hisPerfRecoTPinvptForEff_;
  map<string, TH1F*> hisPerfRecoTPptForEff_;
  map<string, TH1F*> hisPerfRecoTPetaForEff_;
  //  
  map<string, TH1F*> hisRecoTPd0ForEff_;
  map<string, TH1F*> hisRecoTPz0ForEff_;
  //
  map<string, TH1F*> hisRecoTPinvptForAlgEff_;
  map<string, TH1F*> hisRecoTPptForAlgEff_;
  map<string, TH1F*> hisRecoTPetaForAlgEff_;
  map<string, TH1F*> hisRecoTPphiForAlgEff_;
  //
  map<string, TH1F*> hisPerfRecoTPinvptForAlgEff_;
  map<string, TH1F*> hisPerfRecoTPptForAlgEff_;
  map<string, TH1F*> hisPerfRecoTPetaForAlgEff_;
  //
  map<string, TH1F*> hisRecoTPd0ForAlgEff_;
  map<string, TH1F*> hisRecoTPz0ForAlgEff_;
  //
  map<string, TH1F*> hisRecoTPphisecForAlgEff_;
  map<string, TH1F*> hisPerfRecoTPphisecForAlgEff_;
  map<string, TH1F*> hisRecoTPetasecForAlgEff_;
  map<string, TH1F*> hisPerfRecoTPetasecForAlgEff_;

  map<string, TH1F*> hisRecoTPinvptForAlgEff_inJetPtG30_;
  map<string, TH1F*> hisRecoTPinvptForAlgEff_inJetPtG100_;
  map<string, TH1F*> hisRecoTPinvptForAlgEff_inJetPtG200_;

  // Histograms for track fitting evaluation, where map index specifies name of track fitting algorithm used.

  map<string, TProfile*> profNumFitTracks_;
  map<string, TH1F*> hisNumFitTrks_;
  map<string, TH1F*> hisNumFitTrksPerNon_;
  map<string, TH1F*> hisNumFitTrksPerSect_;

  map<string, TH1F*>     hisStubsPerFitTrack_;
  map<string, TProfile*> profStubsOnFitTracks_;

  map<string, TH1F*> hisFitQinvPtMatched_;
  map<string, TH1F*> hisFitPhi0Matched_;
  map<string, TH1F*> hisFitD0Matched_;
  map<string, TH1F*> hisFitZ0Matched_;
  map<string, TH1F*> hisFitEtaMatched_;

  map<string, TH1F*> hisFitQinvPtUnmatched_;
  map<string, TH1F*> hisFitPhi0Unmatched_;
  map<string, TH1F*> hisFitD0Unmatched_;
  map<string, TH1F*> hisFitZ0Unmatched_;
  map<string, TH1F*> hisFitEtaUnmatched_;

  map<string, TH1F*> hisKalmanNumUpdateCalls_;
  map<string, TH1F*> hisKalmanChi2DofSkipLay0Matched_;
  map<string, TH1F*> hisKalmanChi2DofSkipLay1Matched_;
  map<string, TH1F*> hisKalmanChi2DofSkipLay2Matched_;
  map<string, TH1F*> hisKalmanChi2DofSkipLay0Unmatched_;
  map<string, TH1F*> hisKalmanChi2DofSkipLay1Unmatched_;
  map<string, TH1F*> hisKalmanChi2DofSkipLay2Unmatched_;

  map<string, TH1F*> hisFitChi2Matched_;
  map<string, TH1F*> hisFitChi2DofMatched_;
  map<string, TH1F*> hisFitChi2DofRphiMatched_;
  map<string, TH1F*> hisFitChi2DofRzMatched_;
  map<string, TH1F*> hisFitBeamChi2Matched_;
  map<string, TH1F*> hisFitBeamChi2DofMatched_;
  map<string, TProfile*> profFitChi2VsEtaMatched_;
  map<string, TProfile*> profFitChi2DofVsEtaMatched_;
  map<string, TProfile*> profFitChi2VsInvPtMatched_;
  map<string, TProfile*> profFitChi2DofVsInvPtMatched_;
  map<string, TProfile*> profFitChi2VsTrueD0Matched_;
  map<string, TProfile*> profFitChi2DofVsTrueD0Matched_;
  map<string, TH1F*> hisFitChi2PerfMatched_;
  map<string, TH1F*> hisFitChi2DofPerfMatched_;

  map<string, TH1F*> hisFitChi2Unmatched_;
  map<string, TH1F*> hisFitChi2DofUnmatched_;
  map<string, TH1F*> hisFitChi2DofRphiUnmatched_;
  map<string, TH1F*> hisFitChi2DofRzUnmatched_;
  map<string, TH1F*> hisFitBeamChi2Unmatched_;
  map<string, TH1F*> hisFitBeamChi2DofUnmatched_;
  map<string, TProfile*> profFitChi2VsEtaUnmatched_;
  map<string, TProfile*> profFitChi2DofVsEtaUnmatched_;
  map<string, TProfile*> profFitChi2VsInvPtUnmatched_;
  map<string, TProfile*> profFitChi2DofVsInvPtUnmatched_;

  map<string, TProfile*> profFitChi2VsPurity_;
  map<string, TProfile*> profFitChi2DofVsPurity_;

  map<string, TH1F*> hisDeltaPhitruePSbarrel_;
  map<string, TH1F*> hisDeltaRorZtruePSbarrel_;
  map<string, TH1F*> hisDeltaPhitrue2Sbarrel_;
  map<string, TH1F*> hisDeltaRorZtrue2Sbarrel_;
  map<string, TH1F*> hisDeltaPhitruePSendcap_;
  map<string, TH1F*> hisDeltaRorZtruePSendcap_;
  map<string, TH1F*> hisDeltaPhitrue2Sendcap_;
  map<string, TH1F*> hisDeltaRorZtrue2Sendcap_;
  map<string, TH1F*> hisDeltaPhifakePSbarrel_;
  map<string, TH1F*> hisDeltaRorZfakePSbarrel_;
  map<string, TH1F*> hisDeltaPhifake2Sbarrel_;
  map<string, TH1F*> hisDeltaRorZfake2Sbarrel_;
  map<string, TH1F*> hisDeltaPhifakePSendcap_;
  map<string, TH1F*> hisDeltaRorZfakePSendcap_;
  map<string, TH1F*> hisDeltaPhifake2Sendcap_;
  map<string, TH1F*> hisDeltaRorZfake2Sendcap_;
  map<string, TProfile*> profRecalcRphiChi2VsEtaTrue1_;
  map<string, TProfile*> profRecalcRzChi2VsEtaTrue1_;
  map<string, TProfile*> profRecalcChi2VsEtaTrue1_;
  map<string, TProfile*> profRecalcChi2VsEtaTrue2_;
  map<string, TProfile*> profNsigmaPhivsInvPt_;
  map<string, TProfile*> profNsigmaPhivsR_;
  map<string, TProfile*> profNsigmaPhivsTanl_;

  map<string, TH2F*> hisFitVsSeedQinvPtMatched_;
  map<string, TH2F*> hisFitVsSeedPhi0Matched_;
  map<string, TH2F*> hisFitVsSeedD0Matched_;
  map<string, TH2F*> hisFitVsSeedZ0Matched_;
  map<string, TH2F*> hisFitVsSeedEtaMatched_;

  map<string, TH2F*> hisFitVsSeedQinvPtUnmatched_;
  map<string, TH2F*> hisFitVsSeedPhi0Unmatched_;
  map<string, TH2F*> hisFitVsSeedD0Unmatched_;
  map<string, TH2F*> hisFitVsSeedZ0Unmatched_;
  map<string, TH2F*> hisFitVsSeedEtaUnmatched_;

  map<string, TH2F*>     hisNumStubsVsPurityMatched_;
  map<string, TProfile*> profFitFracTrueStubsVsLayerMatched_;
  map<string, TProfile*> profFitFracTrueStubsVsEtaMatched_;

  map<string, TH2F*> hisFitVsTrueQinvPt_;
  map<string, TH2F*> hisFitVsTruePhi0_;
  map<string, TH2F*> hisFitVsTrueD0_;
  map<string, TH2F*> hisFitVsTrueZ0_;
  map<string, TH2F*> hisFitVsTrueEta_;

  map<string, TH1F*> hisFitQinvPtRes_;
  map<string, TH1F*> hisFitPhi0Res_;
  map<string, TH1F*> hisFitD0Res_;
  map<string, TH1F*> hisFitZ0Res_;
  map<string, TH1F*> hisFitEtaRes_;  

  map<string, TProfile*> hisQoverPtResVsTrueEta_;
  map<string, TProfile*> hisPhi0ResVsTrueEta_;
  map<string, TProfile*> hisEtaResVsTrueEta_;
  map<string, TProfile*> hisZ0ResVsTrueEta_;
  map<string, TProfile*> hisD0ResVsTrueEta_;

  map<string, TProfile*> hisQoverPtResVsTrueInvPt_;
  map<string, TProfile*> hisPhi0ResVsTrueInvPt_;
  map<string, TProfile*> hisEtaResVsTrueInvPt_;
  map<string, TProfile*> hisZ0ResVsTrueInvPt_;
  map<string, TProfile*> hisD0ResVsTrueInvPt_;

  map<string, TProfile*> hisQoverPtResBeamVsTrueEta_;
  map<string, TProfile*> hisPhi0ResBeamVsTrueEta_;
  map<string, TProfile*> hisQoverPtResBeamVsTrueInvPt_;
  map<string, TProfile*> hisPhi0ResBeamVsTrueInvPt_;

  map<string, TH2F*> hisFitEfficiencyVsChi2Dof_;
  map<string, TH2F*> hisNumStubsVsChi2Dof_;
  map<string, TH2F*> hisNumLayersVsChi2Dof_;
  map<string, TH2F*> hisAvgNumStubsPerLayerVsChi2Dof_;

  map<string, TProfile*> profDupFitTrksVsEta_;
  map<string, TProfile*> profDupFitTrksVsInvPt_;

  // Histograms used for efficiency plots made with fitted tracks.
  map<string, TH1F*> hisFitTPinvptForEff_;
  map<string, TH1F*> hisFitTPptForEff_;
  map<string, TH1F*> hisFitTPetaForEff_;
  map<string, TH1F*> hisFitTPphiForEff_;
  map<string, TH1F*> hisPerfFitTPinvptForEff_;
  map<string, TH1F*> hisPerfFitTPptForEff_;
  map<string, TH1F*> hisPerfFitTPetaForEff_;
  map<string, TH1F*> hisFitTPd0ForEff_;
  map<string, TH1F*> hisFitTPz0ForEff_;
  map<string, TH1F*> hisFitTPinvptForAlgEff_;
  map<string, TH1F*> hisFitTPptForAlgEff_;
  map<string, TH1F*> hisFitTPetaForAlgEff_;
  map<string, TH1F*> hisFitTPphiForAlgEff_;
  map<string, TH1F*> hisPerfFitTPinvptForAlgEff_;
  map<string, TH1F*> hisPerfFitTPptForAlgEff_;
  map<string, TH1F*> hisPerfFitTPetaForAlgEff_;
  map<string, TH1F*> hisFitTPd0ForAlgEff_;
  map<string, TH1F*> hisFitTPz0ForAlgEff_;
  map<string, TH1F*> hisFitTPphisecForAlgEff_;
  map<string, TH1F*> hisFitTPetasecForAlgEff_;
  map<string, TH1F*> hisPerfFitTPphisecForAlgEff_;
  map<string, TH1F*> hisPerfFitTPetasecForAlgEff_;
  map<string, TH1F*> hisPerfFitTPinvptForAlgEff_inJetPtG30_;
  map<string, TH1F*> hisPerfFitTPinvptForAlgEff_inJetPtG100_;
  map<string, TH1F*> hisPerfFitTPinvptForAlgEff_inJetPtG200_;

  // Histograms of tracking efficiency & fake rate after Hough transform or after r-z track filter.
  map<string, TEfficiency*> teffEffVsInvPt_;
  map<string, TEfficiency*> teffEffVsPt_;
  map<string, TEfficiency*> teffEffVsEta_;
  map<string, TEfficiency*> teffEffVsPhi_;
  //
  map<string, TEfficiency*> teffPerfEffVsInvPt_;
  map<string, TEfficiency*> teffPerfEffVsPt_;
  map<string, TEfficiency*> teffPerfEffVsEta_;
  //
  map<string, TEfficiency*> teffEffVsD0_;
  map<string, TEfficiency*> teffEffVsZ0_;
  //
  map<string, TEfficiency*> teffAlgEffVsInvPt_;
  map<string, TEfficiency*> teffAlgEffVsPt_;
  map<string, TEfficiency*> teffAlgEffVsEta_;
  map<string, TEfficiency*> teffAlgEffVsPhi_;
  map<string, TEfficiency*> teffAlgEffVsInvPt_inJetPtG30_;
  map<string, TEfficiency*> teffAlgEffVsInvPt_inJetPtG100_;
  map<string, TEfficiency*> teffAlgEffVsInvPt_inJetPtG200_;
  //
  map<string, TEfficiency*> teffPerfAlgEffVsInvPt_;
  map<string, TEfficiency*> teffPerfAlgEffVsPt_;
  map<string, TEfficiency*> teffPerfAlgEffVsEta_;
  //
  map<string, TEfficiency*> teffAlgEffVsD0_;
  map<string, TEfficiency*> teffAlgEffVsZ0_;
  //
  map<string, TEfficiency*> teffAlgEffVsPhiSec_;
  map<string, TEfficiency*> teffAlgEffVsEtaSec_;
  map<string, TEfficiency*> teffPerfAlgEffVsPhiSec_;
  map<string, TEfficiency*> teffPerfAlgEffVsEtaSec_;

  // Histograms of tracking efficiency & fake rate after Hough transform based on tracks after the track fit.
  map<string, TEfficiency*> teffEffFitVsInvPt_;
  map<string, TEfficiency*> teffEffFitVsPt_;
  map<string, TEfficiency*> teffEffFitVsEta_;
  map<string, TEfficiency*> teffEffFitVsPhi_;
  //
  map<string, TEfficiency*> teffPerfEffFitVsInvPt_;
  map<string, TEfficiency*> teffPerfEffFitVsPt_;
  map<string, TEfficiency*> teffPerfEffFitVsEta_;
  //
  map<string, TEfficiency*> teffEffFitVsD0_;
  map<string, TEfficiency*> teffEffFitVsZ0_;
  //
  map<string, TEfficiency*> teffAlgEffFitVsInvPt_;
  map<string, TEfficiency*> teffAlgEffFitVsPt_;
  map<string, TEfficiency*> teffAlgEffFitVsEta_;
  map<string, TEfficiency*> teffAlgEffFitVsPhi_;
  //
  map<string, TEfficiency*> teffPerfAlgEffFitVsInvPt_;
  map<string, TEfficiency*> teffPerfAlgEffFitVsPt_;
  map<string, TEfficiency*> teffPerfAlgEffFitVsEta_;
  map<string, TEfficiency*> teffPerfAlgEffFitVsInvPt_inJetPtG30_;
  map<string, TEfficiency*> teffPerfAlgEffFitVsInvPt_inJetPtG100_;
  map<string, TEfficiency*> teffPerfAlgEffFitVsInvPt_inJetPtG200_;
  //
  map<string, TEfficiency*> teffAlgEffFitVsD0_;
  map<string, TEfficiency*> teffAlgEffFitVsZ0_;
  //
  map<string, TEfficiency*> teffAlgEffFitVsPhiSec_;
  map<string, TEfficiency*> teffAlgEffFitVsEtaSec_;
  map<string, TEfficiency*> teffPerfAlgEffFitVsPhiSec_;
  map<string, TEfficiency*> teffPerfAlgEffFitVsEtaSec_;

  bool plotFirst_;

  // Number of genuine reconstructed and perfectly reconstructed tracks which were fitted.
  map<string, unsigned int> numFitAlgEff_;
  map<string, unsigned int> numFitPerfAlgEff_;

  // Number of genuine reconstructed and perfectly reconstructed tracks which were fitted post-cut.
  map<string, unsigned int> numFitAlgEffPass_;
  map<string, unsigned int> numFitPerfAlgEffPass_;

  // Range in r of each barrel layer.
  map<unsigned int, float> mapBarrelLayerMinR_;
  map<unsigned int, float> mapBarrelLayerMaxR_;
  // Range in z of each endcap wheel.
  map<unsigned int, float> mapEndcapWheelMinZ_;
  map<unsigned int, float> mapEndcapWheelMaxZ_;

  // Range in (r,z) of each module type.
  map<unsigned int, float> mapModuleTypeMinR_;
  map<unsigned int, float> mapModuleTypeMaxR_;
  map<unsigned int, float> mapModuleTypeMinZ_;
  map<unsigned int, float> mapModuleTypeMaxZ_;
  // Extra maps for wierd barrel layers 1-2 & endcap wheels 3-5.
  map<unsigned int, float> mapExtraAModuleTypeMinR_;
  map<unsigned int, float> mapExtraAModuleTypeMaxR_;
  map<unsigned int, float> mapExtraAModuleTypeMinZ_;
  map<unsigned int, float> mapExtraAModuleTypeMaxZ_;
  map<unsigned int, float> mapExtraBModuleTypeMinR_;
  map<unsigned int, float> mapExtraBModuleTypeMaxR_;
  map<unsigned int, float> mapExtraBModuleTypeMinZ_;
  map<unsigned int, float> mapExtraBModuleTypeMaxZ_;
  map<unsigned int, float> mapExtraCModuleTypeMinR_;
  map<unsigned int, float> mapExtraCModuleTypeMaxR_;
  map<unsigned int, float> mapExtraCModuleTypeMinZ_;
  map<unsigned int, float> mapExtraCModuleTypeMaxZ_;
  map<unsigned int, float> mapExtraDModuleTypeMinR_;
  map<unsigned int, float> mapExtraDModuleTypeMaxR_;
  map<unsigned int, float> mapExtraDModuleTypeMinZ_;
  map<unsigned int, float> mapExtraDModuleTypeMaxZ_;

  bool bApproxMistake_;
};

}
#endif
