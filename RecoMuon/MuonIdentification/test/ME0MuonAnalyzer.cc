#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 
#include "FWCore/Utilities/interface/InputTag.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/MuonReco/interface/EmulatedME0Segment.h>
#include <DataFormats/MuonReco/interface/EmulatedME0SegmentCollection.h>

#include <DataFormats/MuonReco/interface/ME0Muon.h>
#include <DataFormats/MuonReco/interface/ME0MuonCollection.h>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "TLorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "DataFormats/Math/interface/deltaR.h"
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

//Associator for chi2: Including header files
#include "SimTracker/TrackAssociatorProducers/plugins/TrackAssociatorByChi2Impl.h"
#include "SimTracker/TrackAssociatorProducers/plugins/TrackAssociatorByHitsImpl.h"

//#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"

#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"

#include "RecoMuon/MuonIdentification/plugins/ME0MuonSelector.cc"

#include "Fit/FitResult.h"
#include "TF1.h" 


#include "TMath.h"
#include "TLorentzVector.h"

#include "TH1.h" 
#include <TH2.h>
#include "TFile.h"
#include <TProfile.h>
#include "TStyle.h"
#include <TCanvas.h>
#include <TLatex.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>


#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TGraph.h"

#include <sstream>    
#include <string>

#include <iostream>
#include <fstream>
#include <sys/stat.h>

class ME0MuonAnalyzer : public edm::EDAnalyzer {
public:
  explicit ME0MuonAnalyzer(const edm::ParameterSet&);
  ~ME0MuonAnalyzer();
  FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
			     int , const AlgebraicSymMatrix66& ,
			     const MagneticField* );

  FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
			     int , const AlgebraicSymMatrix55& ,
			     const MagneticField* );
    void getFromFTS(const FreeTrajectoryState& ,
		  GlobalVector& , GlobalVector& , 
		  int& , AlgebraicSymMatrix66& );


  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void beginRun(edm::Run const&, edm::EventSetup const&);
  void endRun(edm::Run const&, edm::EventSetup const&);

  //protected:
  
  private:

  edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticlesToken_;
  edm::EDGetTokenT <reco::TrackCollection > generalTracksToken_;
  edm::EDGetTokenT <ME0MuonCollection > OurMuonsToken_;
  edm::EDGetTokenT<ME0SegmentCollection> OurSegmentsToken_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track> > > track_Collection_Token;


  bool UseAssociators;
  bool RejectEndcapMuons;
  const TrackAssociatorByChi2Impl* associatorByChi2;



  std::vector<std::string> associators;
  std::vector<edm::InputTag> label;
  std::vector<const reco::TrackToTrackingParticleAssociator*> associator;

  //Histos for plotting
  TString histoFolder;
  TString me0MuonSelector;
  TFile* histoFile; 
  TH1F *Candidate_Eta;  TH1F *Mass_h; 
  TH1F *Segment_Eta;    TH1F *Segment_Phi;    TH1F *Segment_R;  TH2F *Segment_Pos;  
  TH1F *Rechit_Eta;    TH1F *Rechit_Phi;    TH1F *Rechit_R;  TH2F *Rechit_Pos;  
  TH1F *GenMuon_Phi;    TH1F *GenMuon_R;  TH2F *GenMuon_Pos;  
  TH1F *Track_Eta; TH1F *Track_Pt;  TH1F *ME0Muon_Eta; TH1F *ME0Muon_Pt;  TH1F *CheckME0Muon_Eta; 
  TH1F *ME0Muon_SmallBins_Pt;   TH1F *ME0Muon_VariableBins_Pt;
  TH1F *ME0Muon_Cuts_Eta_5_10;   TH1F *ME0Muon_Cuts_Eta_9_11; TH1F *ME0Muon_Cuts_Eta_10_50; TH1F *ME0Muon_Cuts_Eta_50_100; TH1F *ME0Muon_Cuts_Eta_100; 
  TH1F *UnmatchedME0Muon_Eta; TH1F *UnmatchedME0Muon_Pt;    TH1F *UnmatchedME0Muon_Window_Pt;    TH1F *Chi2UnmatchedME0Muon_Eta; 
  TH1F *Chi2UnmatchedME0Muon_Pt; TH1F *Chi2UnmatchedME0Muon_SmallBins_Pt;   TH1F *Chi2UnmatchedME0Muon_VariableBins_Pt; 
  TH1F *UnmatchedME0Muon_SmallBins_Pt;      TH1F *UnmatchedME0Muon_VariableBins_Pt;   
  TH1F *UnmatchedME0Muon_Cuts_Eta_5_10;    TH1F *UnmatchedME0Muon_Cuts_Eta_9_11;  TH1F *UnmatchedME0Muon_Cuts_Eta_10_50;  TH1F *UnmatchedME0Muon_Cuts_Eta_50_100;  TH1F *UnmatchedME0Muon_Cuts_Eta_100;
  TH1F *TracksPerSegment_h;  TH2F *TracksPerSegment_s;  TProfile *TracksPerSegment_p;
  TH2F *ClosestDelR_s; TProfile *ClosestDelR_p;
  TH2F *PtDiff_s; TProfile *PtDiff_p; TH1F *PtDiff_h; TH1F *QOverPtDiff_h; TH1F *PtDiff_rms; TH1F *PtDiff_gaus_narrow; TH1F *PtDiff_gaus_wide;
  TH2F *StandalonePtDiff_s; TProfile *StandalonePtDiff_p; TH1F *StandalonePtDiff_h; TH1F *StandaloneQOverPtDiff_h; TH1F *StandalonePtDiff_rms; TH1F *StandalonePtDiff_gaus_narrow; TH1F *StandalonePtDiff_gaus_wide;
  TH1F *PtDiff_gaus_5_10;  TH1F *PtDiff_gaus_10_50;  TH1F *PtDiff_gaus_50_100; TH1F *PtDiff_gaus_100;
  TH1F *StandalonePtDiff_gaus;
  TH1F *VertexDiff_h;
  TH2F *PDiff_s; TProfile *PDiff_p; TH1F *PDiff_h;
  TH2F *PtDiff_s_5_10;    TH2F *PtDiff_s_10_50;    TH2F *PtDiff_s_50_100;    TH2F *PtDiff_s_100;
  TH1F *FakeTracksPerSegment_h;  TH2F *FakeTracksPerSegment_s;  TProfile *FakeTracksPerSegment_p;
  TH1F *FakeTracksPerAssociatedSegment_h;  TH2F *FakeTracksPerAssociatedSegment_s;  TProfile *FakeTracksPerAssociatedSegment_p;
  TH1F *GenMuon_Eta; TH1F *GenMuon_Pt;   TH1F *MatchedME0Muon_Eta; TH1F *MatchedME0Muon_Pt; TH1F *Chi2MatchedME0Muon_Eta; TH1F *Chi2MatchedME0Muon_Pt; 
  TH1F *GenMuon_SmallBins_Pt;  TH1F *MatchedME0Muon_SmallBins_Pt; TH1F *Chi2MatchedME0Muons_Pt; TH1F *Chi2MatchedME0Muon_SmallBins_Pt; 
  TH1F *GenMuon_VariableBins_Pt;  TH1F *MatchedME0Muon_VariableBins_Pt; TH1F *Chi2MatchedME0Muon_VariableBins_Pt; 
  TH1F *TPMuon_Eta;   TH1F *TPMuon_SmallBins_Pt;    TH1F *TPMuon_Pt; TH1F *TPMuon_VariableBins_Pt;  
  TH1F *MatchedME0Muon_Eta_5_10;    TH1F *MatchedME0Muon_Eta_9_11;  TH1F *MatchedME0Muon_Eta_10_50;  TH1F *MatchedME0Muon_Eta_50_100;  TH1F *MatchedME0Muon_Eta_100;
  TH1F *Chi2MatchedME0Muon_Eta_5_10;   TH1F *Chi2MatchedME0Muon_Eta_9_11; TH1F *Chi2MatchedME0Muon_Eta_10_50;  TH1F *Chi2MatchedME0Muon_Eta_50_100;  TH1F *Chi2MatchedME0Muon_Eta_100;
  TH1F *GenMuon_Eta_5_10;   TH1F *GenMuon_Eta_9_11;  TH1F *GenMuon_Eta_10_50;  TH1F *GenMuon_Eta_50_100;  TH1F *GenMuon_Eta_100;
  TH1F *MuonRecoEff_Eta;  TH1F *MuonRecoEff_Pt;   TH1F *Chi2MuonRecoEff_Eta;  
  TH1F *MuonRecoEff_Eta_5_10;   TH1F *MuonRecoEff_Eta_9_11;  TH1F *MuonRecoEff_Eta_10_50;  TH1F *MuonRecoEff_Eta_50_100;  TH1F *MuonRecoEff_Eta_100;
  TH1F *Chi2MuonRecoEff_Eta_5_10;    TH1F *Chi2MuonRecoEff_Eta_9_11;  TH1F *Chi2MuonRecoEff_Eta_10_50;  TH1F *Chi2MuonRecoEff_Eta_50_100;  TH1F *Chi2MuonRecoEff_Eta_100;
  TH1F *FakeRate_Eta;  TH1F *FakeRate_Pt;  TH1F *FakeRate_Eta_PerEvent;    TH1F *Chi2FakeRate_Eta;  

  TH1F *Chi2FakeRate_WideBinning_Eta;  
  TH1F *Chi2FakeRate_WidestBinning_Eta;  
  TH1F *FakeRate_WideBinning_Eta;
  TH1F *FakeRate_WidestBinning_Eta;
  TH1F *UnmatchedME0Muon_Cuts_WideBinning_Eta;
  TH1F *UnmatchedME0Muon_Cuts_WidestBinning_Eta;
  TH1F *ME0Muon_Cuts_WideBinning_Eta; 
  TH1F *ME0Muon_Cuts_WidestBinning_Eta;
  TH1F *Chi2UnmatchedME0Muon_WideBinning_Eta; 
  TH1F *Chi2UnmatchedME0Muon_WidestBinning_Eta; 
  TH1F *TPMuon_WideBinning_Eta;
  TH1F *TPMuon_WidestBinning_Eta;
  TH1F *GenMuon_WideBinning_Eta;
  TH1F *GenMuon_WidestBinning_Eta;
  TH1F *MatchedME0Muon_WideBinning_Eta;
  TH1F *MatchedME0Muon_WidestBinning_Eta;
  TH1F *Chi2MatchedME0Muon_WideBinning_Eta;
  TH1F *Chi2MatchedME0Muon_WidestBinning_Eta;
  TH1F *MuonRecoEff_WideBinning_Eta;
  TH1F *MuonRecoEff_WidestBinning_Eta;
  TH1F *Chi2MuonRecoEff_WideBinning_Eta;  
  TH1F *Chi2MuonRecoEff_WidestBinning_Eta;  


  TH1F *FakeRate_Eta_5_10;    TH1F *FakeRate_Eta_9_11;  TH1F *FakeRate_Eta_10_50;  TH1F *FakeRate_Eta_50_100;  TH1F *FakeRate_Eta_100;
  TH1F *MuonAllTracksEff_Eta;  TH1F *MuonAllTracksEff_Pt;
  TH1F *MuonUnmatchedTracksEff_Eta;  TH1F *MuonUnmatchedTracksEff_Pt; TH1F *FractionMatched_Eta;

  TH1F *StandaloneMuonRecoEff_Eta;   TH1F *StandaloneMuonRecoEff_WideBinning_Eta;   TH1F *StandaloneMuonRecoEff_WidestBinning_Eta;
  TH1F *UnmatchedME0Muon_Cuts_Eta;TH1F *ME0Muon_Cuts_Eta;
  TH1F *StandaloneMatchedME0Muon_Eta;    TH1F *StandaloneMatchedME0Muon_WideBinning_Eta;    TH1F *StandaloneMatchedME0Muon_WidestBinning_Eta;
  TH1F *DelR_Segment_GenMuon;

  TH1F *SegPosDirPhiDiff_True_h;    TH1F *SegPosDirEtaDiff_True_h;     TH1F *SegPosDirPhiDiff_All_h;    TH1F *SegPosDirEtaDiff_All_h;   
  TH1F *SegTrackDirPhiDiff_True_h;    TH1F *SegTrackDirEtaDiff_True_h;     TH1F *SegTrackDirPhiDiff_All_h;    TH1F *SegTrackDirEtaDiff_All_h;   TH1F *SegTrackDirPhiPull_True_h;   TH1F *SegTrackDirPhiPull_All_h;   

  TH1F *SegGenDirPhiDiff_True_h;    TH1F *SegGenDirEtaDiff_True_h;     TH1F *SegGenDirPhiDiff_All_h;    TH1F *SegGenDirEtaDiff_All_h;   TH1F *SegGenDirPhiPull_True_h;   TH1F *SegGenDirPhiPull_All_h;   

  TH1F *XDiff_h;   TH1F *YDiff_h;   TH1F *XPull_h;   TH1F *YPull_h;


  TH1F *DelR_Window_Under5; TH1F  *Pt_Window_Under5;
  TH1F *DelR_Track_Window_Under5; TH1F  *Pt_Track_Window_Under5;  TH1F  *Pt_Track_Window;
  TH1F *DelR_Track_Window_Failed_Under5; TH1F  *Pt_Track_Window_Failed_Under5;  TH1F  *Pt_Track_Window_Failed;

  TH1F *FailedTrack_Window_XPull;    TH1F *FailedTrack_Window_YPull;    TH1F *FailedTrack_Window_PhiDiff;
  TH1F *FailedTrack_Window_XDiff;    TH1F *FailedTrack_Window_YDiff;    

  TH1F *NormChi2_h;    TH1F *NormChi2Prob_h; TH2F *NormChi2VsHits_h;	TH2F *chi2_vs_eta_h;  TH1F *AssociatedChi2_h;  TH1F *AssociatedChi2_Prob_h;

  TH1F *PreMatch_TP_R;   TH1F *PostMatch_TP_R;  TH1F *PostMatch_BX0_TP_R;

  TH2F *UnmatchedME0Muon_ScatterPlot;

  double  FakeRatePtCut, MatchingWindowDelR;

  double Nevents;

  TH1F *Nevents_h;

};

ME0MuonAnalyzer::ME0MuonAnalyzer(const edm::ParameterSet& iConfig) 
{
  histoFile = new TFile(iConfig.getParameter<std::string>("HistoFile").c_str(), "recreate");
  histoFolder = iConfig.getParameter<std::string>("HistoFolder").c_str();
  me0MuonSelector = iConfig.getParameter<std::string>("ME0MuonSelectionType").c_str();
  RejectEndcapMuons = iConfig.getParameter< bool >("RejectEndcapMuons");
  UseAssociators = iConfig.getParameter< bool >("UseAssociators");

  FakeRatePtCut   = iConfig.getParameter<double>("FakeRatePtCut");
  MatchingWindowDelR   = iConfig.getParameter<double>("MatchingWindowDelR");

  //Associator for chi2: getting parameters
  UseAssociators = iConfig.getParameter< bool >("UseAssociators");
  associators = iConfig.getParameter< std::vector<std::string> >("associators");

  label = iConfig.getParameter< std::vector<edm::InputTag> >("label");
  edm::InputTag genParticlesTag ("genParticles");
  genParticlesToken_ = consumes<reco::GenParticleCollection>(genParticlesTag);
  edm::InputTag trackingParticlesTag ("mix","MergedTrackTruth");
  trackingParticlesToken_ = consumes<TrackingParticleCollection>(trackingParticlesTag);
  edm::InputTag generalTracksTag ("generalTracks");
  generalTracksToken_ = consumes<reco::TrackCollection>(generalTracksTag);
  edm::InputTag OurMuonsTag ("me0SegmentMatching");
  OurMuonsToken_ = consumes<ME0MuonCollection>(OurMuonsTag);
  edm::InputTag OurSegmentsTag ("me0Segments");
  OurSegmentsToken_ = consumes<ME0SegmentCollection>(OurSegmentsTag);

  //Getting tokens and doing consumers for track associators
  for (unsigned int www=0;www<label.size();www++){
    track_Collection_Token.push_back(consumes<edm::View<reco::Track> >(label[www]));
  }

  if (UseAssociators) {
    for (auto const& thisassociator :associators) {
      consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag(thisassociator));
    }
  }


  std::cout<<"Contructor end"<<std::endl;
}



void ME0MuonAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {

  //Making the directory to write plot pngs to
  mkdir(histoFolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  //Histos for plotting
  Candidate_Eta = new TH1F("Candidate_Eta"      , "Candidate #eta"   , 4, 2.0, 2.8 );

  Nevents_h = new TH1F("Nevents_h"      , "Nevents"   , 2, 0, 2 );

  Track_Eta = new TH1F("Track_Eta"      , "Track #eta"   , 4, 2.0, 2.8 );
  Track_Pt = new TH1F("Track_Pt"      , "Muon p_{T}"   , 120,0 , 120. );

  Segment_Eta = new TH1F("Segment_Eta"      , "Segment #eta"   , 4, 2.0, 2.8 );
  Segment_Phi = new TH1F("Segment_Phi"      , "Segment #phi"   , 60, -3, 3. );
  Segment_R = new TH1F("Segment_R"      , "Segment r"   , 30, 0, 150 );
  Segment_Pos = new TH2F("Segment_Pos"      , "Segment x,y"   ,100,-100.,100., 100,-100.,100. );

  Rechit_Eta = new TH1F("Rechit_Eta"      , "Rechit #eta"   , 4, 2.0, 2.8 );
  Rechit_Phi = new TH1F("Rechit_Phi"      , "Rechit #phi"   , 60, -3, 3. );
  Rechit_R = new TH1F("Rechit_R"      , "Rechit r"   , 30, 0, 150 );
  Rechit_Pos = new TH2F("Rechit_Pos"      , "Rechit x,y"   ,100,-100.,100., 100,-100.,100. );

  GenMuon_Phi = new TH1F("GenMuon_Phi"      , "GenMuon #phi"   , 60, -3, 3. );
  GenMuon_R = new TH1F("GenMuon_R"      , "GenMuon r"   , 30, 0, 150 );
  GenMuon_Pos = new TH2F("GenMuon_Pos"      , "GenMuon x,y"   ,100,-100.,100., 100,-100.,100. );

  ME0Muon_Eta = new TH1F("ME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_5_10 = new TH1F("ME0Muon_Cuts_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_9_11 = new TH1F("ME0Muon_Cuts_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_10_50 = new TH1F("ME0Muon_Cuts_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_50_100 = new TH1F("ME0Muon_Cuts_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_100 = new TH1F("ME0Muon_Cuts_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.8 );

  CheckME0Muon_Eta = new TH1F("CheckME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Pt = new TH1F("ME0Muon_Pt"      , "Muon p_{T}"   , 120,0 , 120. );

  GenMuon_Eta = new TH1F("GenMuon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_5_10 = new TH1F("GenMuon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_9_11 = new TH1F("GenMuon_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_10_50 = new TH1F("GenMuon_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_50_100 = new TH1F("GenMuon_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_100 = new TH1F("GenMuon_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.8 );

  TPMuon_Eta = new TH1F("TPMuon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );

  GenMuon_Pt = new TH1F("GenMuon_Pt"      , "Muon p_{T}"   , 100,0 , 100. );

  //Default variable binning scheme
  double varbins[]={0.,5.,10.,15.,20.,25.,30.,40.,50.,70.,100.};
  GenMuon_SmallBins_Pt = new TH1F("GenMuon_SmallBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  GenMuon_VariableBins_Pt = new TH1F("GenMuon_VariableBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  TPMuon_Pt = new TH1F("TPMuon_Pt"      , "Muon p_{T}"   , 100,0 , 100. );
  TPMuon_SmallBins_Pt = new TH1F("TPMuon_SmallBins_Pt"      , "Muon p_{T}"   ,10,varbins);
  TPMuon_VariableBins_Pt = new TH1F("TPMuon_VariableBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  MatchedME0Muon_Eta = new TH1F("MatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  StandaloneMatchedME0Muon_Eta = new TH1F("StandaloneMatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  StandaloneMatchedME0Muon_WideBinning_Eta = new TH1F("StandaloneMatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  StandaloneMatchedME0Muon_WidestBinning_Eta = new TH1F("StandaloneMatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );
  MatchedME0Muon_Eta_5_10 = new TH1F("MatchedME0Muon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_9_11 = new TH1F("MatchedME0Muon_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_10_50 = new TH1F("MatchedME0Muon_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_50_100 = new TH1F("MatchedME0Muon_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_100 = new TH1F("MatchedME0Muon_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.8 );


  Chi2MatchedME0Muon_Eta_5_10 = new TH1F("Chi2MatchedME0Muon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  Chi2MatchedME0Muon_Eta_9_11 = new TH1F("Chi2MatchedME0Muon_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.8 );
  Chi2MatchedME0Muon_Eta_10_50 = new TH1F("Chi2MatchedME0Muon_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.8 );
  Chi2MatchedME0Muon_Eta_50_100 = new TH1F("Chi2MatchedME0Muon_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.8 );
  Chi2MatchedME0Muon_Eta_100 = new TH1F("Chi2MatchedME0Muon_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.8 );

  MatchedME0Muon_Pt = new TH1F("MatchedME0Muon_Pt"      , "Muon p_{T}"   , 100,0 , 100. );
  MatchedME0Muon_SmallBins_Pt = new TH1F("MatchedME0Muon_SmallBins_Pt"      , "Muon p_{T}"   ,10,varbins);
  MatchedME0Muon_VariableBins_Pt = new TH1F("MatchedME0Muon_VariableBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  ME0Muon_SmallBins_Pt = new TH1F("ME0Muon_SmallBins_Pt"      , "Muon p_{T}"   ,10,varbins);
  ME0Muon_VariableBins_Pt = new TH1F("ME0Muon_VariableBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  Chi2MatchedME0Muon_Eta = new TH1F("Chi2MatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  Chi2MatchedME0Muon_Pt = new TH1F("Chi2MatchedME0Muon_Pt"      , "Muon p_{T}"   , 100,0 , 100. );
  Chi2MatchedME0Muon_SmallBins_Pt = new TH1F("Chi2MatchedME0Muon_SmallBins_Pt"      , "Muon p_{T}"   ,10,varbins);
  Chi2MatchedME0Muon_VariableBins_Pt = new TH1F("Chi2MatchedME0Muon_VariableBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  Chi2UnmatchedME0Muon_Eta = new TH1F("Chi2UnmatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );

  UnmatchedME0Muon_Eta = new TH1F("UnmatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_5_10 = new TH1F("UnmatchedME0Muon_Cuts_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_9_11 = new TH1F("UnmatchedME0Muon_Cuts_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_10_50 = new TH1F("UnmatchedME0Muon_Cuts_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_50_100 = new TH1F("UnmatchedME0Muon_Cuts_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_100 = new TH1F("UnmatchedME0Muon_Cuts_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.8 );

  UnmatchedME0Muon_Pt = new TH1F("UnmatchedME0Muon_Pt"      , "Muon p_{T}"   , 100,0 , 100. );
  UnmatchedME0Muon_SmallBins_Pt = new TH1F("UnmatchedME0Muon_SmallBins_Pt"      , "Muon p_{T}"   ,10,varbins);
  UnmatchedME0Muon_VariableBins_Pt = new TH1F("UnmatchedME0Muon_VariableBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  Chi2UnmatchedME0Muon_Pt = new TH1F("Chi2UnmatchedME0Muon_Pt"      , "Muon p_{T}"   , 100,0 , 100. );
  Chi2UnmatchedME0Muon_SmallBins_Pt = new TH1F("Chi2UnmatchedME0Muon_SmallBins_Pt"      , "Muon p_{T}"   ,10,varbins);
  Chi2UnmatchedME0Muon_VariableBins_Pt = new TH1F("Chi2UnmatchedME0Muon_VariableBins_Pt"      , "Muon p_{T}"   ,10,varbins);

  UnmatchedME0Muon_Window_Pt = new TH1F("UnmatchedME0Muon_Window_Pt"      , "Muon p_{T}"   , 500,0 , 50 );

  UnmatchedME0Muon_Cuts_Eta = new TH1F("UnmatchedME0Muon_Cuts_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta = new TH1F("ME0Muon_Cuts_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );

  Mass_h = new TH1F("Mass_h"      , "Mass"   , 100, 0., 200 );

  MuonRecoEff_Eta = new TH1F("MuonRecoEff_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );

  MuonRecoEff_Eta_5_10 = new TH1F("MuonRecoEff_Eta_5_10"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Eta_9_11 = new TH1F("MuonRecoEff_Eta_9_11"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Eta_10_50 = new TH1F("MuonRecoEff_Eta_10_50"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Eta_50_100 = new TH1F("MuonRecoEff_Eta_50_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Eta_100 = new TH1F("MuonRecoEff_Eta_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  Chi2MuonRecoEff_Eta = new TH1F("Chi2MuonRecoEff_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  Chi2MuonRecoEff_Eta_5_10 = new TH1F("Chi2MuonRecoEff_Eta_5_10"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  Chi2MuonRecoEff_Eta_9_11 = new TH1F("Chi2MuonRecoEff_Eta_9_11"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  Chi2MuonRecoEff_Eta_10_50 = new TH1F("Chi2MuonRecoEff_Eta_10_50"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  Chi2MuonRecoEff_Eta_50_100 = new TH1F("Chi2MuonRecoEff_Eta_50_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  Chi2MuonRecoEff_Eta_100 = new TH1F("Chi2MuonRecoEff_Eta_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );

  MuonRecoEff_Pt = new TH1F("MuonRecoEff_Pt"      , "Fraction of ME0Muons matched to gen muons"   ,8, 0,40  );

  StandaloneMuonRecoEff_Eta = new TH1F("StandaloneMuonRecoEff_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,4, 2.0, 2.8  );
  StandaloneMuonRecoEff_WideBinning_Eta = new TH1F("StandaloneMuonRecoEff_WideBinning_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,8, 2.0, 2.8  );
  StandaloneMuonRecoEff_WidestBinning_Eta = new TH1F("StandaloneMuonRecoEff_WidestBinning_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,16, 2.0, 2.8  );

  FakeRate_Eta = new TH1F("FakeRate_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_5_10 = new TH1F("FakeRate_Eta_5_10"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_9_11 = new TH1F("FakeRate_Eta_9_11"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_10_50 = new TH1F("FakeRate_Eta_10_50"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_50_100 = new TH1F("FakeRate_Eta_50_100"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_100 = new TH1F("FakeRate_Eta_100"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );

  Chi2FakeRate_Eta = new TH1F("Chi2FakeRate_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );

  FakeRate_Eta_PerEvent = new TH1F("FakeRate_Eta_PerEvent"      , "PU140, unmatched ME0Muons/all ME0Muons normalized by N_{events}"   ,4, 2.0, 2.8  );
  FakeRate_Pt = new TH1F("FakeRate_Pt"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,8, 0,40  );

  MuonAllTracksEff_Eta = new TH1F("MuonAllTracksEff_Eta"      , "All ME0Muons over all tracks"   ,4, 2.0, 2.8  );
  MuonAllTracksEff_Pt = new TH1F("MuonAllTracksEff_Pt"      , "All ME0Muons over all tracks"   ,8, 0,40  );

  MuonUnmatchedTracksEff_Eta = new TH1F("MuonUnmatchedTracksEff_Eta"      , "Unmatched ME0Muons over all ME0Muons"   ,4, 2.0, 2.8  );
  MuonUnmatchedTracksEff_Pt = new TH1F("MuonUnmatchedTracksEff_Pt"      , "Unmatched ME0Muons over all ME0Muons"   ,8, 0,40  );

  TracksPerSegment_h = new TH1F("TracksPerSegment_h", "Number of tracks", 60,0.,60.);
  TracksPerSegment_s = new TH2F("TracksPerSegment_s" , "Tracks per segment vs |#eta|", 4, 2.0, 2.8, 60,0.,60.);
  TracksPerSegment_p = new TProfile("TracksPerSegment_p" , "Tracks per segment vs |#eta|", 4, 2.0, 2.8, 0.,60.);

  FakeTracksPerSegment_h = new TH1F("FakeTracksPerSegment_h", "Number of fake tracks", 60,0.,60.);
  FakeTracksPerSegment_s = new TH2F("FakeTracksPerSegment_s" , "Fake tracks per segment", 10, 2.4, 4.0, 100,0.,60.);
  FakeTracksPerSegment_p = new TProfile("FakeTracksPerSegment_p" , "Average N_{tracks}/segment not matched to genmuons", 10, 2.4, 4.0, 0.,60.);

  FakeTracksPerAssociatedSegment_h = new TH1F("FakeTracksPerAssociatedSegment_h", "Number of fake tracks", 60,0.,60.);
  FakeTracksPerAssociatedSegment_s = new TH2F("FakeTracksPerAssociatedSegment_s" , "Fake tracks per segment", 10, 2.4, 4.0, 100,0.,60.);
  FakeTracksPerAssociatedSegment_p = new TProfile("FakeTracksPerAssociatedSegment_p" , "Average N_{tracks}/segment not matched to genmuons", 10, 2.4, 4.0, 0.,60.);

  ClosestDelR_s = new TH2F("ClosestDelR_s" , "#Delta R", 4, 2.0, 2.8, 15,0.,0.15);
  ClosestDelR_p = new TProfile("ClosestDelR_p" , "#Delta R", 4, 2.0, 2.8, 0.,0.15);

  DelR_Window_Under5 = new TH1F("DelR_Window_Under5","#Delta R", 15, 0,0.15  );
  Pt_Window_Under5 = new TH1F("Pt_Window_Under5","pt",500, 0,50  );

  DelR_Track_Window_Under5 = new TH1F("DelR_Track_Window_Under5","#Delta R", 15, 0,0.15  );
  Pt_Track_Window_Under5 = new TH1F("Pt_Track_Window_Under5","pt",20, 0,5  );
  Pt_Track_Window = new TH1F("Pt_Track_Window","pt",500, 0,  50);

  DelR_Track_Window_Failed_Under5 = new TH1F("DelR_Track_Window_Failed_Under5","#Delta R", 15, 0,0.15  );
  Pt_Track_Window_Failed_Under5 = new TH1F("Pt_Track_Window_Failed_Under5","pt",20, 0,5  );
  Pt_Track_Window_Failed = new TH1F("Pt_Track_Window_Failed","pt",500, 0,  50);

  FailedTrack_Window_XPull = new TH1F("FailedTrack_Window_XPull", "X Pull failed tracks", 100, 0,20);
  FailedTrack_Window_YPull = new TH1F("FailedTrack_Window_YPull", "Y  Pull failed tracks", 100, 0,20);
  FailedTrack_Window_XDiff = new TH1F("FailedTrack_Window_XDiff", "X Diff failed tracks", 100, 0,20);
  FailedTrack_Window_YDiff = new TH1F("FailedTrack_Window_YDiff", "Y  Diff failed tracks", 100, 0,20);

  FailedTrack_Window_PhiDiff = new TH1F("FailedTrack_Window_PhiDiff", "Phi Dir Diff failed tracks", 100,0 ,2.0);

  DelR_Segment_GenMuon = new TH1F("DelR_Segment_GenMuon", "#Delta R between me0segment and gen muon",200,0,2);
  FractionMatched_Eta = new TH1F("FractionMatched_Eta"      , "Fraction of ME0Muons that end up successfully matched (matched/all)"   ,4, 2.0, 2.8  );

  PtDiff_s = new TH2F("PtDiff_s" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);

  PtDiff_s_5_10 = new TH2F("PtDiff_s_5_10" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);
  PtDiff_s_10_50 = new TH2F("PtDiff_s_10_50" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);
  PtDiff_s_50_100 = new TH2F("PtDiff_s_50_100" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);
  PtDiff_s_100 = new TH2F("PtDiff_s_100" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);

  PtDiff_h = new TH1F("PtDiff_h" , "pt resolution", 100,-0.5,0.5);
  QOverPtDiff_h = new TH1F("QOverPtDiff_h" , "q/pt resolution", 100,-0.5,0.5);
  PtDiff_p = new TProfile("PtDiff_p" , "pt resolution vs. #eta", 4, 2.0, 2.8, -1.0,1.0,"s");

  StandalonePtDiff_s = new TH2F("StandalonePtDiff_s" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);
  StandalonePtDiff_h = new TH1F("StandalonePtDiff_h" , "pt resolution", 100,-0.5,0.5);
  StandaloneQOverPtDiff_h = new TH1F("StandaloneQOverPtDiff_h" , "q/pt resolution", 100,-0.5,0.5);
  StandalonePtDiff_p = new TProfile("StandalonePtDiff_p" , "pt resolution vs. #eta", 4, 2.0, 2.8, -1.0,1.0,"s");

  PtDiff_rms    = new TH1F( "PtDiff_rms",    "RMS", 4, 2.0, 2.8 ); 
  PtDiff_gaus_wide    = new TH1F( "PtDiff_gaus_wide",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_narrow    = new TH1F( "PtDiff_gaus_narrow",    "GAUS_NARROW", 4, 2.0, 2.8 ); 

  PtDiff_gaus_5_10    = new TH1F( "PtDiff_gaus_5_10",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_10_50    = new TH1F( "PtDiff_gaus_10_50",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_50_100    = new TH1F( "PtDiff_gaus_50_100",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_100    = new TH1F( "PtDiff_gaus_100",    "GAUS_WIDE", 4, 2.0, 2.8 ); 

  StandalonePtDiff_gaus    = new TH1F( "StandalonePtDiff_gaus",    "GAUS_WIDE", 4, 2.0, 2.8 ); 

  PDiff_s = new TH2F("PDiff_s" , "Relative p difference", 4, 2.0, 2.8, 50,0.,0.5);
  PDiff_h = new TH1F("PDiff_s" , "Relative p difference", 50,0.,0.5);
  PDiff_p = new TProfile("PDiff_p" , "Relative p difference", 4, 2.0, 2.8, 0.,1.0,"s");

  VertexDiff_h = new TH1F("VertexDiff_h", "Difference in vertex Z", 50, 0, 0.2);

  SegPosDirPhiDiff_True_h = new TH1F("SegPosDirPhiDiff_True_h", "#phi Dir. Diff. Real Muons", 50, -2,2);
  SegPosDirEtaDiff_True_h = new TH1F("SegPosDirEtaDiff_True_h", "#eta Dir. Diff. Real Muons", 50, -2,2);

  SegPosDirPhiDiff_All_h = new TH1F("SegPosDirPhiDiff_All_h", "#phi Dir. Diff. All Muons", 50, -3,3);
  SegPosDirEtaDiff_All_h = new TH1F("SegPosDirEtaDiff_All_h", "#eta Dir. Diff. All Muons", 50, -3,3);

  SegTrackDirPhiDiff_True_h = new TH1F("SegTrackDirPhiDiff_True_h", "#phi Dir. Diff. Real Muons", 50, -2,2);
  SegTrackDirEtaDiff_True_h = new TH1F("SegTrackDirEtaDiff_True_h", "#eta Dir. Diff. Real Muons", 50, -2,2);

  SegTrackDirPhiPull_True_h = new TH1F("SegTrackDirPhiPull_True_h", "#phi Dir. Pull. Real Muons", 50, -3,3);
  SegTrackDirPhiPull_All_h = new TH1F("SegTrackDirPhiPull_True_h", "#phi Dir. Pull. All Muons", 50, -3,3);

  SegTrackDirPhiDiff_All_h = new TH1F("SegTrackDirPhiDiff_All_h", "#phi Dir. Diff. All Muons", 50, -3,3);
  SegTrackDirEtaDiff_All_h = new TH1F("SegTrackDirEtaDiff_All_h", "#eta Dir. Diff. All Muons", 50, -3,3);

  SegGenDirPhiDiff_True_h = new TH1F("SegGenDirPhiDiff_True_h", "#phi Dir. Diff. Real Muons", 50, -2,2);
  SegGenDirEtaDiff_True_h = new TH1F("SegGenDirEtaDiff_True_h", "#eta Dir. Diff. Real Muons", 50, -2,2);

  SegGenDirPhiPull_True_h = new TH1F("SegGenDirPhiPull_True_h", "#phi Dir. Pull. Real Muons", 50, -3,3);
  SegGenDirPhiPull_All_h = new TH1F("SegGenDirPhiPull_True_h", "#phi Dir. Pull. All Muons", 50, -3,3);

  SegGenDirPhiDiff_All_h = new TH1F("SegGenDirPhiDiff_All_h", "#phi Dir. Diff. All Muons", 50, -3,3);
  SegGenDirEtaDiff_All_h = new TH1F("SegGenDirEtaDiff_All_h", "#eta Dir. Diff. All Muons", 50, -3,3);


  PreMatch_TP_R = new TH1F("PreMatch_TP_R", "r distance from TP pre match to beamline", 100, 0, 10);
  PostMatch_TP_R = new TH1F("PostMatch_TP_R", "r distance from TP post match to beamline", 200, 0, 20);
  PostMatch_BX0_TP_R = new TH1F("PostMatch_BX0_TP_R", "r distance from TP post match to beamline", 200, 0, 20);


  XDiff_h = new TH1F("XDiff_h", "X Diff", 100, -10.0, 10.0 );
  YDiff_h = new TH1F("YDiff_h", "Y Diff", 100, -50.0, 50.0 ); 
  XPull_h = new TH1F("XPull_h", "X Pull", 100, -5.0, 5.0 );
  YPull_h = new TH1F("YPull_h", "Y Pull", 40, -50.0, 50.0 );

  MuonRecoEff_WideBinning_Eta = new TH1F("MuonRecoEff_WideBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,8, 2.0, 2.8  );
  MuonRecoEff_WidestBinning_Eta = new TH1F("MuonRecoEff_WidestBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,16, 2.0, 2.8  );
  Chi2MuonRecoEff_WideBinning_Eta = new TH1F("Chi2MuonRecoEff_WideBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,8, 2.0, 2.8  );
  Chi2MuonRecoEff_WidestBinning_Eta = new TH1F("Chi2MuonRecoEff_WidestBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,16, 2.0, 2.8  );
  Chi2FakeRate_WideBinning_Eta = new TH1F("Chi2FakeRate_WideBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,8, 2.0, 2.8  );
  Chi2FakeRate_WidestBinning_Eta = new TH1F("Chi2FakeRate_WidestBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,16, 2.0, 2.8  );
  FakeRate_WideBinning_Eta = new TH1F("FakeRate_WideBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,8, 2.0, 2.8  );
  FakeRate_WidestBinning_Eta = new TH1F("FakeRate_WidestBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,16, 2.0, 2.8  );

  UnmatchedME0Muon_Cuts_WideBinning_Eta = new TH1F("UnmatchedME0Muon_Cuts_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_WidestBinning_Eta = new TH1F("UnmatchedME0Muon_Cuts_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );
  ME0Muon_Cuts_WideBinning_Eta = new TH1F("ME0Muon_Cuts_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  ME0Muon_Cuts_WidestBinning_Eta = new TH1F("ME0Muon_Cuts_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );
  Chi2UnmatchedME0Muon_WideBinning_Eta = new TH1F("Chi2UnmatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  Chi2UnmatchedME0Muon_WidestBinning_Eta = new TH1F("Chi2UnmatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );
  TPMuon_WideBinning_Eta = new TH1F("TPMuon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  TPMuon_WidestBinning_Eta = new TH1F("TPMuon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );
  GenMuon_WideBinning_Eta = new TH1F("GenMuon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  GenMuon_WidestBinning_Eta = new TH1F("GenMuon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );
  MatchedME0Muon_WideBinning_Eta = new TH1F("MatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  MatchedME0Muon_WidestBinning_Eta = new TH1F("MatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );
  Chi2MatchedME0Muon_WideBinning_Eta = new TH1F("Chi2MatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.8 );
  Chi2MatchedME0Muon_WidestBinning_Eta = new TH1F("Chi2MatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.8 );

  UnmatchedME0Muon_ScatterPlot = new TH2F("UnmatchedME0Muon_ScatterPlot"      , "Muon #eta vs. #phi"   , 16, 2.0, 2.8, 8, 0., 3.14 );

 
  AssociatedChi2_h = new TH1F("AssociatedChi2_h","Associated #chi^{2}",50,0,50);
  AssociatedChi2_Prob_h = new TH1F("AssociatedChi2_h","Associated #chi^{2}",50,0,1);
  NormChi2_h = new TH1F("NormChi2_h","normalized #chi^{2}", 200, 0, 20);
  NormChi2Prob_h = new TH1F("NormChi2Prob_h","normalized #chi^{2} probability", 100, 0, 1);
  NormChi2VsHits_h = new TH2F("NormChi2VsHits_h","#chi^{2} vs nhits",25,0,25,100,0,10);
  chi2_vs_eta_h = new TH2F("chi2_vs_eta_h","#chi^{2} vs #eta",4, 2.0, 2.8 , 200, 0, 20);

  Nevents=0;


}


ME0MuonAnalyzer::~ME0MuonAnalyzer(){}

void
ME0MuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)

{

  Nevents_h->Fill(1);
  using namespace edm;


  if (UseAssociators) {
    edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
    for (unsigned int w=0;w<associators.size();w++) {
      iEvent.getByLabel(associators[w],theAssociator);
      associator.push_back( theAssociator.product() );
    }
  }


  using namespace reco;
  Handle<GenParticleCollection> genParticles;
  iEvent.getByToken(genParticlesToken_, genParticles);
  const GenParticleCollection genParticlesForChi2 = *(genParticles.product());

  unsigned int gensize=genParticles->size();

  Handle<TrackingParticleCollection> trackingParticles;
  iEvent.getByToken(trackingParticlesToken_, trackingParticles);


  if (RejectEndcapMuons){
    //Section to turn off signal muons in the endcaps, to approximate a nu gun
    for(unsigned int i=0; i<gensize; ++i) {
      const reco::GenParticle& CurrentParticle=(*genParticles)[i];
      if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
	if (fabs(CurrentParticle.eta()) > 1.9 ) {
	  //std::cout<<"Found a signal muon outside the barrel, exiting the function"<<std::endl;
	  return;
	}
      }
    }      
  }



  Nevents++;


  Handle <TrackCollection > generalTracks;
  iEvent.getByToken (generalTracksToken_, generalTracks);

  Handle <std::vector<ME0Muon> > OurMuons;
  iEvent.getByToken (OurMuonsToken_, OurMuons);

  Handle<ME0SegmentCollection> OurSegments;
  iEvent.getByToken(OurSegmentsToken_,OurSegments);


  edm::ESHandle<ME0Geometry> me0Geom;
  iSetup.get<MuonGeometryRecord>().get(me0Geom);

  ESHandle<MagneticField> bField;
  iSetup.get<IdealMagneticFieldRecord>().get(bField);
  ESHandle<Propagator> shProp;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAlong", shProp);
  
  //    -----First, make a vector of bools for each ME0Muon

  std::vector<bool> IsMatched;
  std::vector<int> SegIdForMatch;
  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon){
    if (!muon::isGoodMuon(*thisMuon, muon::Tight)) continue;
    IsMatched.push_back(false);
    SegIdForMatch.push_back(-1);
  }
 

  //=====Finding ME0Muons that match gen muons, plotting the closest of those
  std::vector<int> MatchedSegIds;

  for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
     
      double LowestDelR = 9999;
      double thisDelR = 9999;
      int MatchedID = -1;
      int ME0MuonID = 0;


      std::vector<double> ReferenceTrackPt;

      double VertexDiff=-1,PtDiff=-1,QOverPtDiff=-1,PDiff=-1;

      for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
	   thisMuon != OurMuons->end(); ++thisMuon){
	if (!muon::isGoodMuon(*thisMuon, muon::Tight)) continue;
	TrackRef tkRef = thisMuon->innerTrack();
	SegIdForMatch.push_back(thisMuon->me0segid());
	thisDelR = reco::deltaR(CurrentParticle,*tkRef);
	ReferenceTrackPt.push_back(tkRef->pt());


	if (( tkRef->pt() > FakeRatePtCut )  ){
	  if (thisDelR < MatchingWindowDelR ){
	    if (tkRef->pt() < 5.0){
	      DelR_Window_Under5->Fill(thisDelR);
	      Pt_Window_Under5->Fill(tkRef->pt());
	    }
	    if (thisDelR < LowestDelR){
	      LowestDelR = thisDelR;
	      //if (fabs(tkRef->pt() - CurrentParticle.pt())/CurrentParticle.pt() < 0.50) MatchedID = ME0MuonID;
	      MatchedID = ME0MuonID;
	      VertexDiff = fabs(tkRef->vz()-CurrentParticle.vz());
	      QOverPtDiff = ( (tkRef->charge() /tkRef->pt()) - (CurrentParticle.charge()/CurrentParticle.pt() ) )/  (CurrentParticle.charge()/CurrentParticle.pt() );
	      PtDiff = (tkRef->pt() - CurrentParticle.pt())/CurrentParticle.pt();
	      PDiff = (tkRef->p() - CurrentParticle.p())/CurrentParticle.p();
	    }
	  }
	}
	
	ME0MuonID++;

      }

      for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
	   thisTrack != generalTracks->end();++thisTrack){
	//TrackRef tkRef = thisTrack->innerTrack();
	thisDelR = reco::deltaR(CurrentParticle,*thisTrack);

	if ((thisTrack->pt() > FakeRatePtCut ) ){
	  if (thisDelR < MatchingWindowDelR ){
	    if (thisTrack->pt() < 5.0){
	      DelR_Track_Window_Under5->Fill(thisDelR);
	      Pt_Track_Window_Under5->Fill(thisTrack->pt());
	    }
	    Pt_Track_Window->Fill(thisTrack->pt());
	  }
	}
      }
      if (MatchedID == -1){
	
	for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
	     thisTrack != generalTracks->end();++thisTrack){
	  //TrackRef tkRef = thisTrack->innerTrack();
	  thisDelR = reco::deltaR(CurrentParticle,*thisTrack);


	  if ( (thisTrack->pt() > FakeRatePtCut ) && (TMath::Abs(thisTrack->eta()) < 2.8) && (TMath::Abs(thisTrack->eta()) > 2.0) )  {
	    if (thisDelR < MatchingWindowDelR ){
	      if (thisTrack->pt() < 5.0){
		DelR_Track_Window_Failed_Under5->Fill(thisDelR);
		Pt_Track_Window_Failed_Under5->Fill(thisTrack->pt());
	      }
	      Pt_Track_Window_Failed->Fill(thisTrack->pt());
	    }
	  }
	}
      }

      if (MatchedID != -1){
	IsMatched[MatchedID] = true;

	if ((CurrentParticle.pt() >FakeRatePtCut) ){
	  MatchedME0Muon_Eta->Fill(fabs(CurrentParticle.eta()));
	  if ( (TMath::Abs(CurrentParticle.eta()) > 2.0) && (TMath::Abs(CurrentParticle.eta()) < 2.8) )  {
	    MatchedME0Muon_Pt->Fill(CurrentParticle.pt());
	    MatchedME0Muon_SmallBins_Pt->Fill(CurrentParticle.pt());
	    MatchedME0Muon_VariableBins_Pt->Fill(CurrentParticle.pt());
	  }


	  MatchedME0Muon_WideBinning_Eta->Fill(fabs(CurrentParticle.eta()));
	  MatchedME0Muon_WidestBinning_Eta->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 5.0) && (CurrentParticle.pt() <= 10.0) )  	MatchedME0Muon_Eta_5_10->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 9.0) && (CurrentParticle.pt() <= 11.0) )  	MatchedME0Muon_Eta_9_11->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 10.0) && (CurrentParticle.pt() <= 50.0) )	MatchedME0Muon_Eta_10_50->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 50.0) && (CurrentParticle.pt() <= 100.0) )	MatchedME0Muon_Eta_50_100->Fill(fabs(CurrentParticle.eta()));
	  if ( CurrentParticle.pt() > 100.0) 		MatchedME0Muon_Eta_100->Fill(fabs(CurrentParticle.eta()));
	


	  VertexDiff_h->Fill(VertexDiff);
	  PtDiff_h->Fill(PtDiff);	
	  QOverPtDiff_h->Fill(QOverPtDiff);
	  PtDiff_s->Fill(CurrentParticle.eta(),PtDiff);
	  if ( (CurrentParticle.pt() > 5.0) && (CurrentParticle.pt() <= 10.0) ) 	PtDiff_s_5_10->Fill(CurrentParticle.eta(),PtDiff);
	  if ( (CurrentParticle.pt() > 10.0) && (CurrentParticle.pt() <= 50.0) )	PtDiff_s_10_50->Fill(CurrentParticle.eta(),PtDiff);
	  if ( (CurrentParticle.pt() > 50.0) && (CurrentParticle.pt() <= 100.0) )	PtDiff_s_50_100->Fill(CurrentParticle.eta(),PtDiff);
	  if ( CurrentParticle.pt() > 100.0) 	PtDiff_s_100->Fill(CurrentParticle.eta(),PtDiff);
	  PtDiff_p->Fill(CurrentParticle.eta(),PtDiff);
	
	  PDiff_h->Fill(PDiff);
	  PDiff_s->Fill(CurrentParticle.eta(),PDiff);
	  PDiff_p->Fill(CurrentParticle.eta(),PDiff);
	}
	MatchedSegIds.push_back(SegIdForMatch[MatchedID]);
      }


	if ( (CurrentParticle.pt() >FakeRatePtCut) ){
	  GenMuon_Eta->Fill(fabs(CurrentParticle.eta()));
	  GenMuon_WideBinning_Eta->Fill(fabs(CurrentParticle.eta()));
	  GenMuon_WidestBinning_Eta->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 5.0) && (CurrentParticle.pt() <= 10.0) )  	GenMuon_Eta_5_10->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 9.0) && (CurrentParticle.pt() <= 11.0) )  	GenMuon_Eta_9_11->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 10.0) && (CurrentParticle.pt() <= 50.0) )	GenMuon_Eta_10_50->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 50.0) && (CurrentParticle.pt() <= 100.0) )	GenMuon_Eta_50_100->Fill(fabs(CurrentParticle.eta()));
	  if ( CurrentParticle.pt() > 100.0) 		GenMuon_Eta_100->Fill(fabs(CurrentParticle.eta()));
	  GenMuon_Phi->Fill(CurrentParticle.phi());
	  if ( (fabs(CurrentParticle.eta()) > 2.0) && (fabs(CurrentParticle.eta()) < 2.8) ) {
	    GenMuon_SmallBins_Pt->Fill(CurrentParticle.pt());
	    GenMuon_VariableBins_Pt->Fill(CurrentParticle.pt());
	    GenMuon_Pt->Fill(CurrentParticle.pt());
	  }
	}

    }
  }



  //Del R study ===========================
  for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  

      double LowestDelR = 9999;
      double thisDelR = 9999;

      for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
	   thisMuon != OurMuons->end(); ++thisMuon){
	if (!muon::isGoodMuon(*thisMuon, muon::Tight)) continue;
	TrackRef tkRef = thisMuon->innerTrack();
	thisDelR = reco::deltaR(CurrentParticle,*tkRef);
	if (thisDelR < LowestDelR) LowestDelR = thisDelR;
      }
    
    ClosestDelR_s->Fill(CurrentParticle.eta(), LowestDelR);
    ClosestDelR_p->Fill(CurrentParticle.eta(), LowestDelR);
    }
  }

  //====================================

  //   -----Finally, we loop over all the ME0Muons in the event
  //   -----Before, we plotted the gen muon pt and eta for the efficiency plot of matches
  //   -----Now, each time a match failed, we plot the ME0Muon pt and eta
  int ME0MuonID = 0;
  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon){
    if (!muon::isGoodMuon(*thisMuon, muon::Tight)) continue;
    TrackRef tkRef = thisMuon->innerTrack();
    //Moved resolution stuff here, only calculate resolutions for matched muons!
    if (!IsMatched[ME0MuonID]){

      UnmatchedME0Muon_Eta->Fill(fabs(tkRef->eta()));

      if ((tkRef->pt() > FakeRatePtCut) && (TMath::Abs(tkRef->eta()) < 2.8) )  {
	if ( (tkRef->pt() > 5.0) && (tkRef->pt() <= 10.0) )  	UnmatchedME0Muon_Cuts_Eta_5_10->Fill(fabs(tkRef->eta()));
	if ( (tkRef->pt() > 9.0) && (tkRef->pt() <= 11.0) )  	UnmatchedME0Muon_Cuts_Eta_9_11->Fill(fabs(tkRef->eta()));
	if ( (tkRef->pt() > 10.0) && (tkRef->pt() <= 20.0) )	UnmatchedME0Muon_Cuts_Eta_10_50->Fill(fabs(tkRef->eta()));
	if ( (tkRef->pt() > 20.0) && (tkRef->pt() <= 40.0) )	UnmatchedME0Muon_Cuts_Eta_50_100->Fill(fabs(tkRef->eta()));
	if ( tkRef->pt() > 40.0) 		UnmatchedME0Muon_Cuts_Eta_100->Fill(fabs(tkRef->eta()));

	UnmatchedME0Muon_Cuts_Eta->Fill(fabs(tkRef->eta()));
	UnmatchedME0Muon_Cuts_WideBinning_Eta->Fill(fabs(tkRef->eta()));
	UnmatchedME0Muon_Cuts_WidestBinning_Eta->Fill(fabs(tkRef->eta()));

	UnmatchedME0Muon_ScatterPlot->Fill(fabs(tkRef->eta()), fabs(tkRef->phi()) );

	for(unsigned int i=0; i<gensize; ++i) {
	  const reco::GenParticle& CurrentParticle=(*genParticles)[i];
	  double thisDelR = reco::deltaR(CurrentParticle,*tkRef);
	  if (thisDelR < MatchingWindowDelR){
	    if ( (TMath::Abs(tkRef->eta()) < 2.8) ) UnmatchedME0Muon_Window_Pt->Fill(tkRef->pt());
	  }
	}
	if ( (TMath::Abs(tkRef->eta()) > 2.0) && (TMath::Abs(tkRef->eta()) < 2.8) ) {
	  UnmatchedME0Muon_Pt->Fill(tkRef->pt());
	  UnmatchedME0Muon_SmallBins_Pt->Fill(tkRef->pt());
	  UnmatchedME0Muon_VariableBins_Pt->Fill(tkRef->pt());
	}
    
      }
    }
    ME0MuonID++;
  }
  


 //Track Association by Chi2 or hits:


  //Map the list of all me0muons that failed or passed delR matching to a list of only Tight me0Muons that failed or passed delR matching
  std::vector<bool> SkimmedIsMatched;
  int i_me0muon=0;
  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon, ++i_me0muon){
    if (!muon::isGoodMuon(*thisMuon, muon::Tight)) continue;
    SkimmedIsMatched.push_back(IsMatched[i_me0muon]);
  }

  //int w=0;
  //std::cout<<"associators size = "<<associators.size()<<std::endl;
  if (UseAssociators) {
    for (unsigned int ww=0;ww<associators.size();ww++){

      for (unsigned int www=0;www<label.size();www++){

	reco::RecoToSimCollection recSimColl;
	reco::SimToRecoCollection simRecColl;
	edm::Handle<View<Track> >  trackCollection;
	
	
	
	unsigned int trackCollectionSize = 0;
      
	if(!iEvent.getByToken(track_Collection_Token[www], trackCollection)){
	  recSimColl.post_insert();
	  simRecColl.post_insert();
	  
	}
	
	else {
	  trackCollectionSize = trackCollection->size();
	  recSimColl=associator[ww]->associateRecoToSim(trackCollection,
							trackingParticles);
	  
	  
	  simRecColl=associator[ww]->associateSimToReco(trackCollection,
						      trackingParticles);
	  
	  
	}
	
	for (TrackingParticleCollection::size_type i=0; i<trackingParticles->size(); i++){
	  
	  const TrackingParticle& TPCheck=(*trackingParticles)[i];
	  if (abs(TPCheck.pdgId()) != 13) continue;
	  
	  TrackingParticleRef tpr(trackingParticles, i);
	  TrackingParticle* tp=const_cast<TrackingParticle*>(tpr.get());
	  TrackingParticle::Vector momentumTP; 
	  TrackingParticle::Point vertexTP;
	  
	  if (abs(tp->pdgId()) != 13) continue;
	  
	  momentumTP = tp->momentum();
	  vertexTP = tp->vertex();
	  
	  //This section fills the denominator for the chi2 efficiency...
	  
	  
	  if ((tp->pt() >FakeRatePtCut) ){
	    bool SignalMuon=false;
	    if (tp->status() !=-99){
	      int motherid=-1;
	      if ((*tp->genParticle_begin())->numberOfMothers()>0)  {
		if ((*tp->genParticle_begin())->mother()->numberOfMothers()>0){
		  motherid=(*tp->genParticle_begin())->mother()->mother()->pdgId();
		}
	      }
	      
  	    std::cout<<"Mother ID = "<<motherid<<std::endl;
	    
  	    if ( 
  		( (tp->status()==1) && ( (*tp->genParticle_begin())->numberOfMothers()==0 ) )  ||
  		( (tp->status()==1) ) ) SignalMuon=true;
	    
  	    //if (SignalMuon) PreMatch_TP_R->Fill( sqrt(pow(tp->vertex().x(),2) + pow(tp->vertex().y(),2)) );
	    }
	    if (SignalMuon){
	      TPMuon_Eta->Fill(fabs(tp->eta()));
	      TPMuon_WideBinning_Eta->Fill(fabs(tp->eta()));
	      TPMuon_WidestBinning_Eta->Fill(fabs(tp->eta()));
	      if ( (fabs(tp->eta()) > 2.0) && (fabs(tp->eta()) < 2.8) ) {
		TPMuon_SmallBins_Pt->Fill(tp->pt());
		TPMuon_VariableBins_Pt->Fill(tp->pt());
		TPMuon_Pt->Fill(tp->pt());
	      }
	      
	    }
	    
	  }
	  if ( (fabs(tp->eta()) > 2.0) && (fabs(tp->eta()) < 2.8) )  PreMatch_TP_R->Fill( sqrt(pow(tp->vertex().x(),2) + pow(tp->vertex().y(),2)) );
  	}// END for (TrackingParticleCollection::size_type i=0; i<trackingParticles->size(); i++){
      
      for(View<Track>::size_type i=0; i<trackCollectionSize; ++i){
  	RefToBase<Track> track(trackCollection, i);

  	std::vector<std::pair<TrackingParticleRef, double> > tp;
  	std::vector<std::pair<TrackingParticleRef, double> > tpforfake;
  	TrackingParticleRef tpr;
  	TrackingParticleRef tprforfake;

  	//Check if the track is associated to any gen particle
  	bool TrackIsEfficient = false;
  	//std::cout<<"About to check first collection"<<std::endl;
  	if(recSimColl.find(track) != recSimColl.end()){
  	  tp = recSimColl[track];

  	  if (tp.size()!=0) {
  	    tpr = tp.begin()->first;

  	    double assocChi2 = -(tp.begin()->second);
	   
  	    //So this track is matched to a gen particle, lets get that gen particle now

  	    if (  (simRecColl.find(tpr) != simRecColl.end())    ){
  	      std::vector<std::pair<RefToBase<Track>, double> > rt;
  	      if  (simRecColl[tpr].size() > 0){
  		rt=simRecColl[tpr];
  		RefToBase<Track> bestrecotrackforeff = rt.begin()->first;
  		//Only fill the efficiency histo if the track found matches up to a gen particle's best choice
  		if ( (bestrecotrackforeff == track ) && (abs(tpr->pdgId()) == 13) ) {
  		  TrackIsEfficient=true;
  		  //This section fills the numerator of the efficiency calculation...
  		  //if ( (track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) )
  		  PostMatch_TP_R->Fill( sqrt(pow(tpr->vertex().x(),2) + pow(tpr->vertex().y(),2)) );
  		  if (tpr->eventId().bunchCrossing()) PostMatch_BX0_TP_R->Fill( sqrt(pow(tpr->vertex().x(),2) + pow(tpr->vertex().y(),2)) );


  		  if ( (tpr->pt() > FakeRatePtCut) )
  		    {

		      
  		      bool SignalMuon=false;

  		      if (tpr->status() !=-99){
  			int motherid=-1;
  			if ((*tpr->genParticle_begin())->numberOfMothers()>0)  {
  			  if ((*tpr->genParticle_begin())->mother()->numberOfMothers()>0){
  			    motherid=(*tpr->genParticle_begin())->mother()->mother()->pdgId();
  			  }
  			}		
  			std::cout<<"Mother ID = "<<motherid<<std::endl;
  			if ( 
  			    ( (tpr->status()==1) && ( (*tpr->genParticle_begin())->numberOfMothers()==0 ) )  ||
  			    ( (tpr->status()==1)  ) )SignalMuon=true;

  		      }
		      if (SignalMuon){
  			Chi2MatchedME0Muon_Eta->Fill(fabs(tpr->eta()));
  			Chi2MatchedME0Muon_WideBinning_Eta->Fill(fabs(tpr->eta()));
  			Chi2MatchedME0Muon_WidestBinning_Eta->Fill(fabs(tpr->eta()));
  			if ( (TMath::Abs(tpr->eta()) > 2.0) && (TMath::Abs(tpr->eta()) < 2.8) ) {
  			  Chi2MatchedME0Muon_Pt->Fill(tpr->pt());
  			  Chi2MatchedME0Muon_SmallBins_Pt->Fill(tpr->pt());
  			  Chi2MatchedME0Muon_VariableBins_Pt->Fill(tpr->pt());
  			}
			
  			if ( (track->pt() > 5.0) && (track->pt() <= 10.0) )  	Chi2MatchedME0Muon_Eta_5_10->Fill(fabs(tpr->eta()));
  			if ( (track->pt() > 9.0) && (track->pt() <= 11.0) )  	Chi2MatchedME0Muon_Eta_9_11->Fill(fabs(tpr->eta()));
  			if ( (track->pt() > 10.0) && (track->pt() <= 50.0) )	Chi2MatchedME0Muon_Eta_10_50->Fill(fabs(tpr->eta()));
  			if ( (track->pt() > 50.0) && (track->pt() <= 100.0) )	Chi2MatchedME0Muon_Eta_50_100->Fill(fabs(tpr->eta()));
  			if ( track->pt() > 100.0) 		Chi2MatchedME0Muon_Eta_100->Fill(fabs(tpr->eta()));
  		      }

  		    }
  		  //...end section

  		  if ( (track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) )AssociatedChi2_h->Fill(assocChi2);
  		  if ( (track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) )AssociatedChi2_Prob_h->Fill(TMath::Prob((assocChi2)*5,5));
  		}
  	      }
  	    }
	
	    
	    
  	  }
  	}
  	//A simple way of measuring fake rate:
  	if (!TrackIsEfficient) {

  	  if ((track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) ) {
  	    Chi2UnmatchedME0Muon_Eta->Fill(fabs(track->eta()));
  	    Chi2UnmatchedME0Muon_WideBinning_Eta->Fill(fabs(track->eta()));
  	    Chi2UnmatchedME0Muon_WidestBinning_Eta->Fill(fabs(track->eta()));
  	    if ( (TMath::Abs(track->eta()) > 2.0) && (TMath::Abs(track->eta()) < 2.8) ) {
  	      Chi2UnmatchedME0Muon_Pt->Fill(track->pt());
  	      Chi2UnmatchedME0Muon_SmallBins_Pt->Fill(track->pt());
  	      Chi2UnmatchedME0Muon_VariableBins_Pt->Fill(track->pt());
  	    }
  	  }

  	}
  	//End checking of Efficient muons
	
	//Deprecated F.R. method, only used for debugging offline.  Commented out now:

  	// //For Fakes --------------------------------------------  here we fill the numerator for the F.R., Chi2UnmatchedME0Muon_Eta
  	// //The denominator is filled elsewhere, just a histo of all the ME0Muon eta values
  	// //It is ME0Muon_Cuts_Eta, so named because it is all ME0Muons passing the selection (also the pT cut)

	
  	// //Check if finding a track associated to a gen particle fails, or if there is no track in the collection at all

  	// if( (recSimColl.find(track) == recSimColl.end() ) || ( recSimColl[track].size() == 0  ) ){
  	//   if (SkimmedIsMatched[i]){
  	//     if ((track->pt() >FakeRatePtCut) ){
  	//       if (tp.size()!=0) std::cout<<"Found an me0muontrack failing chi2 matching: "<<track->pt()<<", "<<track->eta()<<", "<<tp.begin()->second<<std::endl;
  	//     }
  	//   }
  	// }
	
  	// //Its possible that the track is associated to a gen particle, but isn't the best match and would still fail
  	// //In that case, we go to the gen particle...
  	// else if (recSimColl[track].size() > 0){
  	//   tpforfake = recSimColl[track];
  	//   tprforfake=tpforfake.begin()->first;
  	//   //We now have the gen particle, to check


  	//   //If for some crazy reason we can't find the gen particle, that means its a fake
  	//   if (  (simRecColl.find(tprforfake) == simRecColl.end())  ||  (simRecColl[tprforfake].size() == 0)  ) {
  	//     //Check if this muon matched via Del-R matching
  	//     if (SkimmedIsMatched[i]){
  	//       if ((track->pt() >FakeRatePtCut) ) {
  	// 	if (tp.size()!=0) std::cout<<"Found an me0muontrack failing chi2 matching: "<<track->pt()<<", "<<track->eta()<<", "<<tp.begin()->second<<std::endl;
  	//       }
  	//     }
  	//   }
  	//   //We can probably find the gen particle
  	//   else if(simRecColl[tprforfake].size() > 0)  {
  	//     //We can now access the best possible track for the gen particle that this track was matched to
  	//     std::vector<std::pair<RefToBase<Track>, double> > rtforfake;
  	//     rtforfake=simRecColl[tprforfake];
	   
  	//     RefToBase<Track> bestrecotrack = rtforfake.begin()->first;
  	//     //if the best reco track is NOT the track that we're looking at, we know we have a fake, that was within the cut, but not the closest
  	//     if (bestrecotrack != track) {
  	//       //Check if this muon matched via Del-R matching
  	//       if (IsMatched[i]){
  	// 	if (tp.size()!=0) std::cout<<"Found an me0muontrack failing chi2 matching: "<<track->pt()<<", "<<track->eta()<<", "<<tp.begin()->second<<std::endl;
  	//       }
  	//     }

  	//   }
  	// }

  	//End For Fakes --------------------------------------------

	
  	if (TMath::Abs(track->eta()) < 2.8) CheckME0Muon_Eta->Fill(fabs(track->eta()));	

  	NormChi2_h->Fill(track->normalizedChi2());
  	NormChi2Prob_h->Fill(TMath::Prob(track->chi2(),(int)track->ndof()));
  	NormChi2VsHits_h->Fill(track->numberOfValidHits(),track->normalizedChi2());


  	chi2_vs_eta_h->Fill((track->eta()),track->normalizedChi2());


      }//END for(View<Track>::size_type i=0; i<trackCollectionSize; ++i){
      }// END for (unsigned int www=0;www<label.size();www++)
    }//END     for (unsigned int ww=0;ww<associators.size();ww++){
  }//END if UseAssociators
  
  for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
       thisTrack != generalTracks->end();++thisTrack){
    Track_Eta->Fill(fabs(thisTrack->eta()));
    if ( (TMath::Abs(thisTrack->eta()) > 2.0) && (TMath::Abs(thisTrack->eta()) < 2.8) ) Track_Pt->Fill(thisTrack->pt());
  }

  
  std::vector<double> SegmentEta, SegmentPhi, SegmentR, SegmentX, SegmentY;
  std::vector<int> Ids;
  std::vector<int> Ids_NonGenMuons;
  std::vector<int> UniqueIdList;
  int TrackID=0;

  int MuID = 0;

  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon){
    if (!muon::isGoodMuon(*thisMuon, muon::Tight)) continue;
    TrackRef tkRef = thisMuon->innerTrack();

    ME0Segment Seg = thisMuon->me0segment();
    ME0DetId id =Seg.me0DetId();
    auto roll = me0Geom->etaPartition(id); 
    auto GlobVect(roll->toGlobal(Seg.localPosition()));

    int SegId=thisMuon->me0segid();

    bool IsNew = true;
    for (unsigned int i =0; i < Ids.size(); i++){
      if (SegId == Ids[i]) IsNew=false;
    }

    if (IsNew) {
      UniqueIdList.push_back(SegId);
      SegmentEta.push_back(GlobVect.eta());
      SegmentPhi.push_back(GlobVect.phi());
      SegmentR.push_back(GlobVect.perp());
      SegmentX.push_back(GlobVect.x());
      SegmentY.push_back(GlobVect.y());
    }
    Ids.push_back(SegId);
    if (!IsMatched[TrackID]) Ids_NonGenMuons.push_back(SegId);

    ME0Muon_Eta->Fill(fabs(tkRef->eta()));

    if ((tkRef->pt() > FakeRatePtCut) && (TMath::Abs(tkRef->eta()) < 2.8)){
      ME0Muon_Cuts_Eta->Fill(fabs(tkRef->eta()));
      ME0Muon_Cuts_WideBinning_Eta->Fill(fabs(tkRef->eta()));
      ME0Muon_Cuts_WidestBinning_Eta->Fill(fabs(tkRef->eta()));
      if ( (tkRef->pt() > 5.0) && (tkRef->pt() <= 10.0) )  	ME0Muon_Cuts_Eta_5_10->Fill(fabs(tkRef->eta()));
      if ( (tkRef->pt() > 9.0) && (tkRef->pt() <= 11.0) )  	ME0Muon_Cuts_Eta_9_11->Fill(fabs(tkRef->eta()));
      if ( (tkRef->pt() > 10.0) && (tkRef->pt() <= 20.0) )	ME0Muon_Cuts_Eta_10_50->Fill(fabs(tkRef->eta()));
      if ( (tkRef->pt() > 20.0) && (tkRef->pt() <= 40.0) )	ME0Muon_Cuts_Eta_50_100->Fill(fabs(tkRef->eta()));
      if ( tkRef->pt() > 40.0) 		ME0Muon_Cuts_Eta_100->Fill(fabs(tkRef->eta()));


      if ( (TMath::Abs(tkRef->eta()) > 2.0) && (TMath::Abs(tkRef->eta()) < 2.8) ) {
	ME0Muon_Pt->Fill(tkRef->pt());
	ME0Muon_SmallBins_Pt->Fill(tkRef->pt());
	ME0Muon_VariableBins_Pt->Fill(tkRef->pt());
      }
    }
    TrackID++;
    MuID++;

  } //END   for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
  

  for (unsigned int i = 0; i < UniqueIdList.size(); i++){
    int Num_Total=0, Num_Fake = 0, Num_Fake_Associated = 0;
    for (unsigned int j = 0; j < Ids.size(); j++){
      if (Ids[j] == UniqueIdList[i]) Num_Total++;
    }

    for (unsigned int j = 0; j < Ids_NonGenMuons.size(); j++){
      if (Ids_NonGenMuons[j] == UniqueIdList[i]) Num_Fake++;
      bool AssociatedWithMatchedSegment = false;
      for (unsigned int isegid=0;isegid < MatchedSegIds.size();isegid++){
	if (MatchedSegIds[isegid]==Ids_NonGenMuons[j]) AssociatedWithMatchedSegment=true;
      }
      if (AssociatedWithMatchedSegment) Num_Fake_Associated++;
    }

    TracksPerSegment_h->Fill((double)Num_Total);
    TracksPerSegment_s->Fill(SegmentEta[i], (double)Num_Total);
    TracksPerSegment_p->Fill(SegmentEta[i], (double)Num_Total);

    FakeTracksPerSegment_h->Fill((double)Num_Fake);
    FakeTracksPerSegment_s->Fill(SegmentEta[i], (double)Num_Fake);
    FakeTracksPerSegment_p->Fill(SegmentEta[i], (double)Num_Fake);

    FakeTracksPerAssociatedSegment_h->Fill((double)Num_Fake_Associated);
    FakeTracksPerAssociatedSegment_s->Fill(SegmentEta[i], (double)Num_Fake_Associated);
    FakeTracksPerAssociatedSegment_p->Fill(SegmentEta[i], (double)Num_Fake_Associated);

  }

  //================  For Segment Plotting
  for (auto thisSegment = OurSegments->begin(); thisSegment != OurSegments->end(); 
       ++thisSegment){
    ME0DetId id = thisSegment->me0DetId();
    auto roll = me0Geom->etaPartition(id); 
    auto GlobVect(roll->toGlobal(thisSegment->localPosition()));
    Segment_Eta->Fill(fabs(GlobVect.eta()));
    Segment_Phi->Fill(GlobVect.phi());
    Segment_R->Fill(GlobVect.perp());
    Segment_Pos->Fill(GlobVect.x(),GlobVect.y());



    auto theseRecHits = thisSegment->specificRecHits();
    
    for (auto thisRecHit = theseRecHits.begin(); thisRecHit!= theseRecHits.end(); thisRecHit++){
      auto me0id = thisRecHit->me0Id();
      auto rollForRechit = me0Geom->etaPartition(me0id);
      
      auto thisRecHitGlobalPoint = rollForRechit->toGlobal(thisRecHit->localPosition()); 
      
      Rechit_Eta->Fill(fabs(thisRecHitGlobalPoint.eta()));
      Rechit_Phi->Fill(thisRecHitGlobalPoint.phi());
      Rechit_R->Fill(thisRecHitGlobalPoint.perp());
      Rechit_Pos->Fill(thisRecHitGlobalPoint.x(),thisRecHitGlobalPoint.y());
      
    }
  }
  //==================

  for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
      double SmallestDelR = 999.;
      for (auto thisSegment = OurSegments->begin(); thisSegment != OurSegments->end(); 
	   ++thisSegment){
	ME0DetId id = thisSegment->me0DetId();
	auto roll = me0Geom->etaPartition(id); 
	  auto GlobVect(roll->toGlobal(thisSegment->localPosition()));
	  if (reco::deltaR(CurrentParticle,GlobVect) < SmallestDelR) SmallestDelR = reco::deltaR(CurrentParticle,GlobVect);
	  
      }
      if ((fabs(CurrentParticle.eta()) < 2.0 ) ||(fabs(CurrentParticle.eta()) > 2.8 )) continue;
      DelR_Segment_GenMuon->Fill(SmallestDelR);
    }
  }
}


void ME0MuonAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) 

{
  
  //Write plots to histo root file and folder
  TString cmsText     = "CMS PhaseII Simulation Prelim.";

  TString lumiText = "PU 140, 14 TeV";
  float cmsTextFont   = 61;  // default is helvetic-bold

 
  //float extraTextFont = 52;  // default is helvetica-italics
  float lumiTextSize     = 0.05;

  float lumiTextOffset   = 0.2;
  float cmsTextSize      = 0.05;
  //float cmsTextOffset    = 0.1;  // only used in outOfFrame version

  // float relPosX    = 0.045;
  // float relPosY    = 0.035;
  // float relExtraDY = 1.2;

  // //ratio of "CMS" and extra text size
  // float extraOverCmsTextSize  = 0.76;



  histoFile->cd();


  TCanvas *c1 = new TCanvas("c1", "canvas" );


  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //setTDRStyle();

  gStyle->SetOptStat(1);     
  //XPull_h->Fit("gaus","","",-1.,1.);
  XPull_h->Draw(); 
  XPull_h->GetXaxis()->SetTitle("Local pulls: X");
  XPull_h->GetXaxis()->SetTitleSize(0.05);
  c1->Print("PullX.png");

  //YPull_h->Fit("gaus");
  YPull_h->Draw(); 
  YPull_h->GetXaxis()->SetTitle("Local pulls: Y");
  YPull_h->GetXaxis()->SetTitleSize(0.05);
  c1->Print("PullY.png");

  gStyle->SetOptStat(1);
  //  XDiff_h->Fit("gaus","","",-1.,1.);
  XDiff_h->Draw(); 
  XDiff_h->GetXaxis()->SetTitle("Local residuals : X");
  XDiff_h->GetXaxis()->SetTitleSize(0.05);
  c1->Print("DiffX.png");

  //  YDiff_h->Fit("gaus");
  YDiff_h->Draw(); 
  YDiff_h->GetXaxis()->SetTitle("Local residuals : Y");
  YDiff_h->GetXaxis()->SetTitleSize(0.05);
  c1->Print("DiffY.png");

  gStyle->SetOptStat(0);
  Nevents_h->Write();

  SegPosDirPhiDiff_True_h->Write();   SegPosDirPhiDiff_True_h->Draw();  c1->Print(histoFolder+"/SegPosDirPhiDiff_True_h.png");
  SegPosDirEtaDiff_True_h->Write();   SegPosDirEtaDiff_True_h->Draw();  c1->Print(histoFolder+"/SegPosDirEtaDiff_True_h.png");

  c1->SetLogy();
  SegPosDirPhiDiff_All_h->Write();   SegPosDirPhiDiff_All_h->Draw();  c1->Print(histoFolder+"/SegPosDirPhiDiff_All_h.png");
  c1->SetLogy();
  SegPosDirEtaDiff_All_h->Write();   SegPosDirEtaDiff_All_h->Draw();  c1->Print(histoFolder+"/SegPosDirEtaDiff_All_h.png");

  SegTrackDirPhiDiff_True_h->Write();   SegTrackDirPhiDiff_True_h->Draw();  c1->Print(histoFolder+"/SegTrackDirPhiDiff_True_h.png");
  SegTrackDirEtaDiff_True_h->Write();   SegTrackDirEtaDiff_True_h->Draw();  c1->Print(histoFolder+"/SegTrackDirEtaDiff_True_h.png");

  SegTrackDirPhiPull_True_h->Write();   SegTrackDirPhiPull_True_h->Draw();  c1->Print(histoFolder+"/SegTrackDirPhiPull_True_h.png");
  SegTrackDirPhiPull_All_h->Write();   SegTrackDirPhiPull_All_h->Draw();  c1->Print(histoFolder+"/SegTrackDirPhiPull_All_h.png");

  c1->SetLogy();
  SegTrackDirPhiDiff_All_h->Write();   SegTrackDirPhiDiff_All_h->Draw();  c1->Print(histoFolder+"/SegTrackDirPhiDiff_All_h.png");
  c1->SetLogy();
  SegTrackDirEtaDiff_All_h->Write();   SegTrackDirEtaDiff_All_h->Draw();  c1->Print(histoFolder+"/SegTrackDirEtaDiff_All_h.png");


  SegGenDirPhiDiff_True_h->Write();   SegGenDirPhiDiff_True_h->Draw();  c1->Print(histoFolder+"/SegGenDirPhiDiff_True_h.png");
  SegGenDirEtaDiff_True_h->Write();   SegGenDirEtaDiff_True_h->Draw();  c1->Print(histoFolder+"/SegGenDirEtaDiff_True_h.png");

  SegGenDirPhiPull_True_h->Write();   SegGenDirPhiPull_True_h->Draw();  c1->Print(histoFolder+"/SegGenDirPhiPull_True_h.png");
  SegGenDirPhiPull_All_h->Write();   SegGenDirPhiPull_All_h->Draw();  c1->Print(histoFolder+"/SegGenDirPhiPull_All_h.png");

  c1->SetLogy();
  SegGenDirPhiDiff_All_h->Write();   SegGenDirPhiDiff_All_h->Draw();  c1->Print(histoFolder+"/SegGenDirPhiDiff_All_h.png");
  c1->SetLogy();
  SegGenDirEtaDiff_All_h->Write();   SegGenDirEtaDiff_All_h->Draw();  c1->Print(histoFolder+"/SegGenDirEtaDiff_All_h.png");

  Candidate_Eta->Write();   Candidate_Eta->Draw();  c1->Print(histoFolder+"/Candidate_Eta.png");
  Track_Eta->Write();   Track_Eta->Draw();  c1->Print(histoFolder+"/Track_Eta.png");
  Track_Pt->Write();   Track_Pt->Draw();  c1->Print(histoFolder+"/Track_Pt.png");

  Segment_Eta->GetXaxis()->SetTitle("me0segment |#eta|");
  Segment_Eta->GetYaxis()->SetTitle(" Num. Segments");
  Segment_Eta->Write();   Segment_Eta->Draw();  
  //GenMuon_Eta->SetLineColor(2);GenMuon_Eta->Draw("SAME"); 
  c1->Print(histoFolder+"/Segment_Eta.png");

  Segment_Phi->Write();   Segment_Phi->Draw();  c1->Print(histoFolder+"/Segment_Phi.png");
  Segment_R->Write();   Segment_R->Draw();  c1->Print(histoFolder+"/Segment_R.png");
  Segment_Pos->Write();   Segment_Pos->Draw();  c1->Print(histoFolder+"/Segment_Pos.png");

  Rechit_Eta->Write();   Rechit_Eta->Draw();   c1->Print(histoFolder+"/Rechit_Eta.png");
  Rechit_Phi->Write();   Rechit_Phi->Draw();  c1->Print(histoFolder+"/Rechit_Phi.png");
  Rechit_R->Write();   Rechit_R->Draw();  c1->Print(histoFolder+"/Rechit_R.png");
  Rechit_Pos->Write();   Rechit_Pos->Draw();  c1->Print(histoFolder+"/Rechit_Pos.png");

  ME0Muon_Eta->Write();   ME0Muon_Eta->Draw();  
  ME0Muon_Eta->GetXaxis()->SetTitle("ME0Muon |#eta|");
  ME0Muon_Eta->GetXaxis()->SetTitleSize(0.05);
  c1->Print(histoFolder+"/ME0Muon_Eta.png");

  CheckME0Muon_Eta->Write();   CheckME0Muon_Eta->Draw();  
  CheckME0Muon_Eta->GetXaxis()->SetTitle("CheckME0Muon |#eta|");
  CheckME0Muon_Eta->GetXaxis()->SetTitleSize(0.05);
  c1->Print(histoFolder+"/CheckME0Muon_Eta.png");

  ME0Muon_Cuts_Eta->Write();   ME0Muon_Cuts_Eta->Draw();  c1->Print(histoFolder+"/ME0Muon_Cuts_Eta.png");
  ME0Muon_Cuts_WidestBinning_Eta->Write();   ME0Muon_Cuts_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/ME0Muon_Cuts_WidestBinning_Eta.png");
  ME0Muon_Cuts_WideBinning_Eta->Write();   ME0Muon_Cuts_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/ME0Muon_Cuts_WideBinning_Eta.png");
  //c1->SetLogy();
  ME0Muon_Pt->Write();   ME0Muon_Pt->Draw();  
  ME0Muon_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  ME0Muon_Pt->GetXaxis()->SetTitleSize(0.05);
  c1->Print(histoFolder+"/ME0Muon_Pt.png");

  ME0Muon_SmallBins_Pt->Write();   ME0Muon_SmallBins_Pt->Draw();  
  ME0Muon_SmallBins_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  ME0Muon_SmallBins_Pt->GetXaxis()->SetTitleSize(0.05);
  c1->Print(histoFolder+"/ME0Muon_SmallBins_Pt.png");

  ME0Muon_VariableBins_Pt->Write();   ME0Muon_VariableBins_Pt->Draw();  
  ME0Muon_VariableBins_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  ME0Muon_VariableBins_Pt->GetXaxis()->SetTitleSize(0.05);
  c1->Print(histoFolder+"/ME0Muon_VariableBins_Pt.png");

  GenMuon_Eta->Write();   GenMuon_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_Eta.png");
  GenMuon_WideBinning_Eta->Write();   GenMuon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_WideBinning_Eta.png");
  GenMuon_WidestBinning_Eta->Write();   GenMuon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_WidestBinning_Eta.png");

  TPMuon_Eta->Write();   TPMuon_Eta->Draw();  c1->Print(histoFolder+"/TPMuon_Eta.png");
  TPMuon_WideBinning_Eta->Write();   TPMuon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/TPMuon_WideBinning_Eta.png");
  TPMuon_WidestBinning_Eta->Write();   TPMuon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/TPMuon_WidestBinning_Eta.png");


  TPMuon_Pt->Write();   TPMuon_Pt->Draw();  c1->Print(histoFolder+"/TPMuon_Pt.png");
  TPMuon_SmallBins_Pt->Write();   TPMuon_SmallBins_Pt->Draw();  c1->Print(histoFolder+"/TPMuon_SmallBins_Pt.png");
  TPMuon_VariableBins_Pt->Write();   TPMuon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/TPMuon_VariableBins_Pt.png");

  GenMuon_Pt->Write();   GenMuon_Pt->Draw();  c1->Print(histoFolder+"/GenMuon_Pt.png");
  GenMuon_SmallBins_Pt->Write();   GenMuon_SmallBins_Pt->Draw();  c1->Print(histoFolder+"/GenMuon_SmallBins_Pt.png");
  GenMuon_VariableBins_Pt->Write();   GenMuon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/GenMuon_VariableBins_Pt.png");

  MatchedME0Muon_Eta->Write();   MatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_Eta.png");
  MatchedME0Muon_WideBinning_Eta->Write();   MatchedME0Muon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_WideBinning_Eta.png");
  MatchedME0Muon_WidestBinning_Eta->Write();   MatchedME0Muon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_WidestBinning_Eta.png");

  StandaloneMatchedME0Muon_Eta->Write();   StandaloneMatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/StandaloneMatchedME0Muon_Eta.png");
  StandaloneMatchedME0Muon_WideBinning_Eta->Write();   StandaloneMatchedME0Muon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/StandaloneMatchedME0Muon_WideBinning_Eta.png");
  StandaloneMatchedME0Muon_WidestBinning_Eta->Write();   StandaloneMatchedME0Muon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/StandaloneMatchedME0Muon_WidestBinning_Eta.png");

  Chi2MatchedME0Muon_Eta->Write();   Chi2MatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/Chi2MatchedME0Muon_Eta.png");
  Chi2MatchedME0Muon_WideBinning_Eta->Write();   Chi2MatchedME0Muon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/Chi2MatchedME0Muon_WideBinning_Eta.png");
  Chi2MatchedME0Muon_WidestBinning_Eta->Write();   Chi2MatchedME0Muon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/Chi2MatchedME0Muon_WidestBinning_Eta.png");
  Chi2UnmatchedME0Muon_Eta->Write();   Chi2UnmatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_Eta.png");
  Chi2UnmatchedME0Muon_WideBinning_Eta->Write();   Chi2UnmatchedME0Muon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_WideBinning_Eta.png");
  Chi2UnmatchedME0Muon_WidestBinning_Eta->Write();   Chi2UnmatchedME0Muon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_WidestBinning_Eta.png");

  gStyle->SetOptStat(1);
  MatchedME0Muon_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  //MatchedME0Muon_Pt->GetYaxis()->SetTitle(" \# of Se");

  MatchedME0Muon_Pt->Write();   MatchedME0Muon_Pt->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_Pt.png");

  MatchedME0Muon_SmallBins_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  //MatchedME0Muon_SmallBins_Pt->GetYaxis()->SetTitle(" \# of Se");

  MatchedME0Muon_SmallBins_Pt->Write();   MatchedME0Muon_SmallBins_Pt->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_SmallBins_Pt.png");

  MatchedME0Muon_VariableBins_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  //MatchedME0Muon_VariableBins_Pt->GetYaxis()->SetTitle(" \# of Se");

  MatchedME0Muon_VariableBins_Pt->Write();   MatchedME0Muon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_VariableBins_Pt.png");
  gStyle->SetOptStat(0);

  UnmatchedME0Muon_Eta->Write();   UnmatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Eta.png");
  UnmatchedME0Muon_Cuts_Eta->Write();   UnmatchedME0Muon_Cuts_Eta->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Cuts_Eta.png");
  UnmatchedME0Muon_Cuts_WideBinning_Eta->Write();   UnmatchedME0Muon_Cuts_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Cuts_WideBinning_Eta.png");
  UnmatchedME0Muon_Cuts_WidestBinning_Eta->Write();   UnmatchedME0Muon_Cuts_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Cuts_WidestBinning_Eta.png");

  UnmatchedME0Muon_ScatterPlot->Write();  UnmatchedME0Muon_ScatterPlot->Draw(); UnmatchedME0Muon_ScatterPlot->Print(histoFolder+"/UnmatchedME0Muon_ScatterPlot.png");


  //gStyle->SetOptStat('oue');
  c1->SetLogy();
  UnmatchedME0Muon_Pt->Write();   UnmatchedME0Muon_Pt->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Pt.png");
  UnmatchedME0Muon_SmallBins_Pt->Write();   UnmatchedME0Muon_SmallBins_Pt->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_SmallBins_Pt.png");
  UnmatchedME0Muon_VariableBins_Pt->Write();   UnmatchedME0Muon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_VariableBins_Pt.png");

  Chi2UnmatchedME0Muon_Pt->Write();   Chi2UnmatchedME0Muon_Pt->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_Pt.png");
  Chi2UnmatchedME0Muon_SmallBins_Pt->Write();   Chi2UnmatchedME0Muon_SmallBins_Pt->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_SmallBins_Pt.png");
  Chi2UnmatchedME0Muon_VariableBins_Pt->Write();   Chi2UnmatchedME0Muon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_VariableBins_Pt.png");


  Chi2MatchedME0Muon_Pt->Write();   Chi2MatchedME0Muon_Pt->Draw();  c1->Print(histoFolder+"/Chi2MatchedME0Muon_Pt.png");
  Chi2MatchedME0Muon_SmallBins_Pt->Write();   Chi2MatchedME0Muon_SmallBins_Pt->Draw();  c1->Print(histoFolder+"/Chi2MatchedME0Muon_SmallBins_Pt.png");
  Chi2MatchedME0Muon_VariableBins_Pt->Write();   Chi2MatchedME0Muon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/Chi2MatchedME0Muon_VariableBins_Pt.png");

  UnmatchedME0Muon_Window_Pt->Write();   UnmatchedME0Muon_Window_Pt->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Window_Pt.png");
  gStyle->SetOptStat(0);

  FailedTrack_Window_XPull->Write();   FailedTrack_Window_XPull->Draw();  c1->Print(histoFolder+"/FailedTrack_Window_XPull.png");
  FailedTrack_Window_YPull->Write();   FailedTrack_Window_YPull->Draw();  c1->Print(histoFolder+"/FailedTrack_Window_YPull.png");
  FailedTrack_Window_XDiff->Write();   FailedTrack_Window_XDiff->Draw();  c1->Print(histoFolder+"/FailedTrack_Window_XDiff.png");
  FailedTrack_Window_YDiff->Write();   FailedTrack_Window_YDiff->Draw();  c1->Print(histoFolder+"/FailedTrack_Window_YDiff.png");
  FailedTrack_Window_PhiDiff->Write();   FailedTrack_Window_PhiDiff->Draw();  c1->Print(histoFolder+"/FailedTrack_Window_PhiDiff.png");

  c1->SetLogy(0);
  TH1F *UnmatchedME0Muon_Cuts_Eta_PerEvent;
  UnmatchedME0Muon_Cuts_Eta_PerEvent = new TH1F("UnmatchedME0Muon_Cuts_Eta_PerEvent"      , "Muon |#eta|"   , 4, 2.0, 2.8 );
  //UnmatchedME0Muon_Cuts_Eta_PerEvent->Sumw2();
  for (int i=1; i<=UnmatchedME0Muon_Cuts_Eta_PerEvent->GetNbinsX(); ++i){
    UnmatchedME0Muon_Cuts_Eta_PerEvent->SetBinContent(i,(UnmatchedME0Muon_Cuts_Eta->GetBinContent(i)));
  }
  UnmatchedME0Muon_Cuts_Eta_PerEvent->Scale(1/Nevents);

  UnmatchedME0Muon_Cuts_Eta_PerEvent->GetXaxis()->SetTitle("ME0Muon |#eta|");
  UnmatchedME0Muon_Cuts_Eta_PerEvent->GetXaxis()->SetTitleSize(0.05);
  
  UnmatchedME0Muon_Cuts_Eta_PerEvent->GetYaxis()->SetTitle("Average Num. ME0Muons per event");
  UnmatchedME0Muon_Cuts_Eta_PerEvent->GetYaxis()->SetTitleSize(0.05);

  UnmatchedME0Muon_Cuts_Eta_PerEvent->Write();   UnmatchedME0Muon_Cuts_Eta_PerEvent->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Cuts_Eta_PerEvent.png");


  TH1F *Chi2UnmatchedME0Muon_Eta_PerEvent;
  Chi2UnmatchedME0Muon_Eta_PerEvent = new TH1F("Chi2UnmatchedME0Muon_Eta_PerEvent"      , "Muon |#eta|"   , 4, 2.0, 2.8 );
  //Chi2UnmatchedME0Muon_Eta_PerEvent->Sumw2();
  for (int i=1; i<=Chi2UnmatchedME0Muon_Eta_PerEvent->GetNbinsX(); ++i){
    Chi2UnmatchedME0Muon_Eta_PerEvent->SetBinContent(i,(Chi2UnmatchedME0Muon_Eta->GetBinContent(i)));
  }
  Chi2UnmatchedME0Muon_Eta_PerEvent->Scale(1/Nevents);

  Chi2UnmatchedME0Muon_Eta_PerEvent->GetXaxis()->SetTitle("ME0Muon |#eta|");
  Chi2UnmatchedME0Muon_Eta_PerEvent->GetXaxis()->SetTitleSize(0.05);
  
  Chi2UnmatchedME0Muon_Eta_PerEvent->GetYaxis()->SetTitle("Average Num. ME0Muons per event");
  Chi2UnmatchedME0Muon_Eta_PerEvent->GetYaxis()->SetTitleSize(0.05);

  Chi2UnmatchedME0Muon_Eta_PerEvent->Write();   Chi2UnmatchedME0Muon_Eta_PerEvent->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_Eta_PerEvent.png");


  TH1F *ME0Muon_Cuts_Eta_PerEvent;
  ME0Muon_Cuts_Eta_PerEvent = new TH1F("ME0Muon_Cuts_Eta_PerEvent"      , "Muon |#eta|"   , 4, 2.0, 2.8 );

  for (int i=1; i<=ME0Muon_Cuts_Eta_PerEvent->GetNbinsX(); ++i){
    ME0Muon_Cuts_Eta_PerEvent->SetBinContent(i,(ME0Muon_Cuts_Eta->GetBinContent(i)));
  }
  ME0Muon_Cuts_Eta_PerEvent->Scale(1/Nevents);

  ME0Muon_Cuts_Eta_PerEvent->Write();   ME0Muon_Cuts_Eta_PerEvent->Draw();  c1->Print(histoFolder+"/ME0Muon_Cuts_Eta_PerEvent.png");

  

  Mass_h->Write();   Mass_h->Draw();  c1->Print(histoFolder+"/Mass_h.png");
  TracksPerSegment_s->SetMarkerStyle(1);
  TracksPerSegment_s->SetMarkerSize(3.0);
  TracksPerSegment_s->Write();     TracksPerSegment_s->Draw();  c1->Print(histoFolder+"/TracksPerSegment_s.png");

  TracksPerSegment_h->Write();     TracksPerSegment_h->Draw();  c1->Print(histoFolder+"/TracksPerSegment_h.png");

  TracksPerSegment_p->GetXaxis()->SetTitle("Gen Muon #eta");
  TracksPerSegment_p->GetYaxis()->SetTitle("Average N_{Tracks} per segment");
  TracksPerSegment_p->Write();     TracksPerSegment_p->Draw();  c1->Print(histoFolder+"/TracksPerSegment_p.png");

  ClosestDelR_s->SetMarkerStyle(1);
  ClosestDelR_s->SetMarkerSize(3.0);
  ClosestDelR_s->Write();     ClosestDelR_s->Draw();  c1->Print(histoFolder+"/ClosestDelR_s.png");

  DelR_Window_Under5->Write();     DelR_Window_Under5->Draw();     c1->Print(histoFolder+"/DelR_Window_Under5.png");
  Pt_Window_Under5->Write();    Pt_Window_Under5->Draw();    c1->Print(histoFolder+"/Pt_Window_Under5.png");

  DelR_Track_Window_Under5->Write();     DelR_Track_Window_Under5->Draw();     c1->Print(histoFolder+"/DelR_Track_Window_Under5.png");
  Pt_Track_Window_Under5->Write();    Pt_Track_Window_Under5->Draw();    c1->Print(histoFolder+"/Pt_Track_Window_Under5.png");
  c1->SetLogy(1);
  Pt_Track_Window->Write();    Pt_Track_Window->Draw();    c1->Print(histoFolder+"/Pt_Track_Window.png");
  c1->SetLogy(0);

  DelR_Track_Window_Failed_Under5->Write();     DelR_Track_Window_Failed_Under5->Draw();     c1->Print(histoFolder+"/DelR_Track_Window_Failed_Under5.png");
  Pt_Track_Window_Failed_Under5->Write();    Pt_Track_Window_Failed_Under5->Draw();    c1->Print(histoFolder+"/Pt_Track_Window_Failed_Under5.png");
  c1->SetLogy(1);
  Pt_Track_Window_Failed->Write();    Pt_Track_Window_Failed->Draw();    c1->Print(histoFolder+"/Pt_Track_Window_Failed.png");
  c1->SetLogy(0);

  DelR_Segment_GenMuon->Write();   DelR_Segment_GenMuon->Draw();  c1->Print(histoFolder+"/DelR_Segment_GenMuon.png");

  ClosestDelR_p->GetXaxis()->SetTitle("Gen Muon #eta");
  ClosestDelR_p->GetYaxis()->SetTitle("Average closest #Delta R track");
  std::cout<<"  ClosestDelR_p values:"<<std::endl;
  for (int i=1; i<=ClosestDelR_p->GetNbinsX(); ++i){
    std::cout<<2.4+(double)i*((4.0-2.4)/40.)<<","<<ClosestDelR_p->GetBinContent(i)<<std::endl;
  }
  ClosestDelR_p->Write();     ClosestDelR_p->Draw();  c1->Print(histoFolder+"/ClosestDelR_p.png");

  FakeTracksPerSegment_s->SetMarkerStyle(1);
  FakeTracksPerSegment_s->SetMarkerSize(3.0);
  FakeTracksPerSegment_s->Write();     FakeTracksPerSegment_s->Draw();  c1->Print(histoFolder+"/FakeTracksPerSegment_s.png");

  FakeTracksPerSegment_h->Write();     FakeTracksPerSegment_h->Draw();  c1->Print(histoFolder+"/FakeTracksPerSegment_h.png");

  FakeTracksPerSegment_p->GetXaxis()->SetTitle("Gen Muon #eta");
  FakeTracksPerSegment_p->GetYaxis()->SetTitle("Average N_{Tracks} per segment");
  FakeTracksPerSegment_p->Write();     FakeTracksPerSegment_p->Draw();  c1->Print(histoFolder+"/FakeTracksPerSegment_p.png");

  FakeTracksPerAssociatedSegment_s->SetMarkerStyle(1);
  FakeTracksPerAssociatedSegment_s->SetMarkerSize(3.0);
  FakeTracksPerAssociatedSegment_s->Write();     FakeTracksPerAssociatedSegment_s->Draw();  c1->Print(histoFolder+"/FakeTracksPerAssociatedSegment_s.png");

  FakeTracksPerAssociatedSegment_h->Write();     FakeTracksPerAssociatedSegment_h->Draw();  c1->Print(histoFolder+"/FakeTracksPerAssociatedSegment_h.png");

  FakeTracksPerAssociatedSegment_p->GetXaxis()->SetTitle("Gen Muon #eta");
  FakeTracksPerAssociatedSegment_p->GetYaxis()->SetTitle("Average N_{Tracks} per segment");
  FakeTracksPerAssociatedSegment_p->Write();     FakeTracksPerAssociatedSegment_p->Draw();  c1->Print(histoFolder+"/FakeTracksPerAssociatedSegment_p.png");

  PreMatch_TP_R->Write(); PreMatch_TP_R->Draw();  c1->Print(histoFolder+"/PreMatch_TP_R.png");
  PostMatch_TP_R->Write(); PostMatch_TP_R->Draw();  c1->Print(histoFolder+"/PostMatch_TP_R.png");
  PostMatch_BX0_TP_R->Write(); PostMatch_BX0_TP_R->Draw();  c1->Print(histoFolder+"/PostMatch_BX0_TP_R.png");

  GenMuon_Eta->Sumw2();  MatchedME0Muon_Eta->Sumw2();  Chi2MatchedME0Muon_Eta->Sumw2();   Chi2UnmatchedME0Muon_Eta->Sumw2();TPMuon_Eta->Sumw2();
  GenMuon_Pt->Sumw2();  MatchedME0Muon_Pt->Sumw2(); MatchedME0Muon_SmallBins_Pt->Sumw2(); MatchedME0Muon_VariableBins_Pt->Sumw2();
  StandaloneMatchedME0Muon_Eta->Sumw2();
  StandaloneMatchedME0Muon_WideBinning_Eta->Sumw2();
  StandaloneMatchedME0Muon_WidestBinning_Eta->Sumw2();
  

  Track_Eta->Sumw2();  ME0Muon_Eta->Sumw2();
  Track_Pt->Sumw2();  ME0Muon_Pt->Sumw2();  ME0Muon_SmallBins_Pt->Sumw2(); ME0Muon_VariableBins_Pt->Sumw2();

  UnmatchedME0Muon_Eta->Sumw2();
  UnmatchedME0Muon_Pt->Sumw2();
  UnmatchedME0Muon_SmallBins_Pt->Sumw2();   UnmatchedME0Muon_VariableBins_Pt->Sumw2();
  
  UnmatchedME0Muon_Cuts_Eta->Sumw2();    ME0Muon_Cuts_Eta->Sumw2();

  ME0Muon_Cuts_Eta_5_10->Sumw2();  ME0Muon_Cuts_Eta_9_11->Sumw2();  ME0Muon_Cuts_Eta_10_50->Sumw2();  ME0Muon_Cuts_Eta_50_100->Sumw2();  ME0Muon_Cuts_Eta_100->Sumw2();
  UnmatchedME0Muon_Cuts_Eta_5_10->Sumw2();    UnmatchedME0Muon_Cuts_Eta_9_11->Sumw2();  UnmatchedME0Muon_Cuts_Eta_10_50->Sumw2();  UnmatchedME0Muon_Cuts_Eta_50_100->Sumw2();  UnmatchedME0Muon_Cuts_Eta_100->Sumw2();
  GenMuon_Eta_5_10->Sumw2();    GenMuon_Eta_9_11->Sumw2();  GenMuon_Eta_10_50->Sumw2();  GenMuon_Eta_50_100->Sumw2();  GenMuon_Eta_100->Sumw2();
  MatchedME0Muon_Eta_5_10->Sumw2();   MatchedME0Muon_Eta_9_11->Sumw2();  MatchedME0Muon_Eta_10_50->Sumw2();  MatchedME0Muon_Eta_50_100->Sumw2();  MatchedME0Muon_Eta_100->Sumw2();

  Chi2MatchedME0Muon_Eta_5_10->Sumw2();   Chi2MatchedME0Muon_Eta_9_11->Sumw2();  Chi2MatchedME0Muon_Eta_10_50->Sumw2();  Chi2MatchedME0Muon_Eta_50_100->Sumw2();  Chi2MatchedME0Muon_Eta_100->Sumw2();

  UnmatchedME0Muon_Cuts_WideBinning_Eta->Sumw2();
  UnmatchedME0Muon_Cuts_WidestBinning_Eta->Sumw2();
  GenMuon_WideBinning_Eta->Sumw2();
  GenMuon_WidestBinning_Eta->Sumw2();
  TPMuon_WideBinning_Eta->Sumw2();
  TPMuon_WidestBinning_Eta->Sumw2();
  MatchedME0Muon_WideBinning_Eta->Sumw2();
  MatchedME0Muon_WidestBinning_Eta->Sumw2();
  Chi2MatchedME0Muon_WideBinning_Eta->Sumw2();
  Chi2MatchedME0Muon_WidestBinning_Eta->Sumw2();
  ME0Muon_Cuts_WideBinning_Eta->Sumw2();
  ME0Muon_Cuts_WidestBinning_Eta->Sumw2();
  Chi2UnmatchedME0Muon_WideBinning_Eta->Sumw2();
  Chi2UnmatchedME0Muon_WidestBinning_Eta->Sumw2();


  //Captions/labels
  std::stringstream PtCutString;

  PtCutString<<"#splitline{DY }{Reco Track p_{T} > "<<FakeRatePtCut<<" GeV}";
  const std::string& ptmp = PtCutString.str();
  const char* pcstr = ptmp.c_str();


  TLatex* txt =new TLatex;
  //txt->SetTextAlign(12);
  //txt->SetTextFont(42);
  txt->SetNDC();
  //txt->SetTextSize(0.05);
  txt->SetTextFont(132);
  txt->SetTextSize(0.05);


  float t = c1->GetTopMargin();
  float r = c1->GetRightMargin();

  TLatex* latex1 = new TLatex;
  latex1->SetNDC();
  latex1->SetTextAngle(0);
  latex1->SetTextColor(kBlack);    


  latex1->SetTextFont(42);
  latex1->SetTextAlign(31); 
  //latex1->SetTextSize(lumiTextSize*t);    
  latex1->SetTextSize(lumiTextSize);    
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  TLatex* latex = new TLatex;
  latex->SetTextFont(cmsTextFont);
  latex->SetNDC();
  latex->SetTextSize(cmsTextSize);
  //latex->SetTextAlign(align_);

  //End captions/labels
  std::cout<<"GenMuon_Eta =  "<<GenMuon_Eta->Integral()<<std::endl;
  std::cout<<"MatchedME0Muon_Eta =  "<<MatchedME0Muon_Eta->Integral()<<std::endl;

  MuonRecoEff_Eta->Divide(MatchedME0Muon_Eta, GenMuon_Eta, 1, 1, "B");
  MuonRecoEff_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta->SetMinimum(MuonRecoEff_Eta->GetMinimum()-0.1);
  MuonRecoEff_Eta->SetMinimum(0);
  //MuonRecoEff_Eta->SetMaximum(MuonRecoEff_Eta->GetMaximum()+0.1);
  MuonRecoEff_Eta->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta->Write();   MuonRecoEff_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta.png");


  MuonRecoEff_WideBinning_Eta->Divide(MatchedME0Muon_WideBinning_Eta, GenMuon_WideBinning_Eta, 1, 1, "B");
  MuonRecoEff_WideBinning_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_WideBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_WideBinning_Eta->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_WideBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_WideBinning_Eta->SetMinimum(MuonRecoEff_WideBinning_Eta->GetMinimum()-0.1);
  MuonRecoEff_WideBinning_Eta->SetMinimum(0);
  //MuonRecoEff_WideBinning_Eta->SetMaximum(MuonRecoEff_WideBinning_Eta->GetMaximum()+0.1);
  MuonRecoEff_WideBinning_Eta->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_WideBinning_Eta->Write();   MuonRecoEff_WideBinning_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_WideBinning_Eta.png");
  c1->Print(histoFolder+"/MuonRecoEff_WideBinning_Eta.png");


  MuonRecoEff_WidestBinning_Eta->Divide(MatchedME0Muon_WidestBinning_Eta, GenMuon_WidestBinning_Eta, 1, 1, "B");
  MuonRecoEff_WidestBinning_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_WidestBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_WidestBinning_Eta->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_WidestBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_WidestBinning_Eta->SetMinimum(MuonRecoEff_WidestBinning_Eta->GetMinimum()-0.1);
  MuonRecoEff_WidestBinning_Eta->SetMinimum(0);
  //MuonRecoEff_WidestBinning_Eta->SetMaximum(MuonRecoEff_WidestBinning_Eta->GetMaximum()+0.1);
  MuonRecoEff_WidestBinning_Eta->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_WidestBinning_Eta->Write();   MuonRecoEff_WidestBinning_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_WidestBinning_Eta.png");
  c1->Print(histoFolder+"/MuonRecoEff_WidestBinning_Eta.png");


  StandaloneMuonRecoEff_Eta->Divide(StandaloneMatchedME0Muon_Eta, GenMuon_Eta, 1, 1, "B");
  StandaloneMuonRecoEff_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  StandaloneMuonRecoEff_Eta->GetXaxis()->SetTitleSize(0.05);
  StandaloneMuonRecoEff_Eta->GetYaxis()->SetTitle("Standalone Muon Efficiency");
  StandaloneMuonRecoEff_Eta->GetYaxis()->SetTitleSize(0.05);
  //StandaloneMuonRecoEff_Eta->SetMinimum(StandaloneMuonRecoEff_Eta->GetMinimum()-0.1);
  StandaloneMuonRecoEff_Eta->SetMinimum(0);
  //StandaloneMuonRecoEff_Eta->SetMaximum(StandaloneMuonRecoEff_Eta->GetMaximum()+0.1);
  StandaloneMuonRecoEff_Eta->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  StandaloneMuonRecoEff_Eta->Write();   StandaloneMuonRecoEff_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestStandaloneMuonRecoEff_Eta.png");
  c1->Print(histoFolder+"/StandaloneMuonRecoEff_Eta.png");


  StandaloneMuonRecoEff_WideBinning_Eta->Divide(StandaloneMatchedME0Muon_WideBinning_Eta, GenMuon_WideBinning_Eta, 1, 1, "B");
  StandaloneMuonRecoEff_WideBinning_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  StandaloneMuonRecoEff_WideBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  StandaloneMuonRecoEff_WideBinning_Eta->GetYaxis()->SetTitle("Standalone Muon Efficiency");
  StandaloneMuonRecoEff_WideBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  //StandaloneMuonRecoEff_WideBinning_Eta->SetMinimum(StandaloneMuonRecoEff_WideBinning_Eta->GetMinimum()-0.1);
  StandaloneMuonRecoEff_WideBinning_Eta->SetMinimum(0);
  //StandaloneMuonRecoEff_WideBinning_Eta->SetMaximum(StandaloneMuonRecoEff_WideBinning_Eta->GetMaximum()+0.1);
  StandaloneMuonRecoEff_WideBinning_Eta->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  StandaloneMuonRecoEff_WideBinning_Eta->Write();   StandaloneMuonRecoEff_WideBinning_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestStandaloneMuonRecoEff_WideBinning_Eta.png");
  c1->Print(histoFolder+"/StandaloneMuonRecoEff_WideBinning_Eta.png");


  StandaloneMuonRecoEff_WidestBinning_Eta->Divide(StandaloneMatchedME0Muon_WidestBinning_Eta, GenMuon_WidestBinning_Eta, 1, 1, "B");
  StandaloneMuonRecoEff_WidestBinning_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  StandaloneMuonRecoEff_WidestBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  StandaloneMuonRecoEff_WidestBinning_Eta->GetYaxis()->SetTitle("Standalone Muon Efficiency");
  StandaloneMuonRecoEff_WidestBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  //StandaloneMuonRecoEff_WidestBinning_Eta->SetMinimum(StandaloneMuonRecoEff_WidestBinning_Eta->GetMinimum()-0.1);
  StandaloneMuonRecoEff_WidestBinning_Eta->SetMinimum(0);
  //StandaloneMuonRecoEff_WidestBinning_Eta->SetMaximum(StandaloneMuonRecoEff_WidestBinning_Eta->GetMaximum()+0.1);
  StandaloneMuonRecoEff_WidestBinning_Eta->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  StandaloneMuonRecoEff_WidestBinning_Eta->Write();   StandaloneMuonRecoEff_WidestBinning_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestStandaloneMuonRecoEff_WidestBinning_Eta.png");
  c1->Print(histoFolder+"/StandaloneMuonRecoEff_WidestBinning_Eta.png");


  MuonRecoEff_Eta_5_10->Divide(MatchedME0Muon_Eta_5_10, GenMuon_Eta_5_10, 1, 1, "B");
  MuonRecoEff_Eta_5_10->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_5_10->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_5_10->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_5_10->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_5_10->SetMinimum(MuonRecoEff_Eta_5_10->GetMinimum()-0.1);
  MuonRecoEff_Eta_5_10->SetMinimum(0);
  //MuonRecoEff_Eta_5_10->SetMaximum(MuonRecoEff_Eta_5_10->GetMaximum()+0.1);
  MuonRecoEff_Eta_5_10->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_5_10->Write();   MuonRecoEff_Eta_5_10->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_5_10.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_5_10.png");

  MuonRecoEff_Eta_9_11->Divide(MatchedME0Muon_Eta_9_11, GenMuon_Eta_9_11, 1, 1, "B");
  MuonRecoEff_Eta_9_11->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_9_11->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_9_11->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_9_11->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_9_11->SetMinimum(MuonRecoEff_Eta_9_11->GetMinimum()-0.1);
  MuonRecoEff_Eta_9_11->SetMinimum(0);
  //MuonRecoEff_Eta_9_11->SetMaximum(MuonRecoEff_Eta_9_11->GetMaximum()+0.1);
  MuonRecoEff_Eta_9_11->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_9_11->Write();   MuonRecoEff_Eta_9_11->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_9_11.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_9_11.png");

  MuonRecoEff_Eta_10_50->Divide(MatchedME0Muon_Eta_10_50, GenMuon_Eta_10_50, 1, 1, "B");
  MuonRecoEff_Eta_10_50->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_10_50->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_10_50->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_10_50->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_10_50->SetMinimum(MuonRecoEff_Eta_10_50->GetMinimum()-0.1);
  MuonRecoEff_Eta_10_50->SetMinimum(0);
  //MuonRecoEff_Eta_10_50->SetMaximum(MuonRecoEff_Eta_10_50->GetMaximum()+0.1);
  MuonRecoEff_Eta_10_50->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_10_50->Write();   MuonRecoEff_Eta_10_50->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_10_50.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_10_50.png");


  MuonRecoEff_Eta_50_100->Divide(MatchedME0Muon_Eta_50_100, GenMuon_Eta_50_100, 1, 1, "B");
  MuonRecoEff_Eta_50_100->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_50_100->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_50_100->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_50_100->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_50_100->SetMinimum(MuonRecoEff_Eta_50_100->GetMinimum()-0.1);
  MuonRecoEff_Eta_50_100->SetMinimum(0);
  //MuonRecoEff_Eta_50_100->SetMaximum(MuonRecoEff_Eta_50_100->GetMaximum()+0.1);
  MuonRecoEff_Eta_50_100->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_50_100->Write();   MuonRecoEff_Eta_50_100->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_50_100.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_50_100.png");


  MuonRecoEff_Eta_100->Divide(MatchedME0Muon_Eta_100, GenMuon_Eta_100, 1, 1, "B");
  MuonRecoEff_Eta_100->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_100->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_100->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_100->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_100->SetMinimum(MuonRecoEff_Eta_100->GetMinimum()-0.1);
  MuonRecoEff_Eta_100->SetMinimum(0);
  //MuonRecoEff_Eta_100->SetMaximum(MuonRecoEff_Eta_100->GetMaximum()+0.1);
  MuonRecoEff_Eta_100->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_100->Write();   MuonRecoEff_Eta_100->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_100.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_100.png");





  Chi2MuonRecoEff_Eta->Divide(Chi2MatchedME0Muon_Eta, TPMuon_Eta, 1, 1, "B");
  std::cout<<"TPMuon_Eta =  "<<TPMuon_Eta->Integral()<<std::endl;
  std::cout<<"Chi2MatchedME0Muon_Eta =  "<<Chi2MatchedME0Muon_Eta->Integral()<<std::endl;
  Chi2MuonRecoEff_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_Eta->GetXaxis()->SetTitleSize(0.05);
  
  Chi2MuonRecoEff_Eta->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_Eta->GetYaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_Eta->SetMinimum(0);
  Chi2MuonRecoEff_Eta->SetMaximum(1.2);

  Chi2MuonRecoEff_Eta->Write();   Chi2MuonRecoEff_Eta->Draw();  

  txt->DrawLatex(0.15,0.2,pcstr);

   
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //latex->SetTextAlign(align_);
  latex->DrawLatex(0.4, 0.85, cmsText);

  c1->Print(histoFolder+"/Chi2MuonRecoEff_Eta.png");

  Chi2MuonRecoEff_WideBinning_Eta->Divide(Chi2MatchedME0Muon_WideBinning_Eta, TPMuon_WideBinning_Eta, 1, 1, "B");
  std::cout<<"TPMuon_WideBinning_Eta =  "<<TPMuon_WideBinning_Eta->Integral()<<std::endl;
  std::cout<<"Chi2MatchedME0Muon_WideBinning_Eta =  "<<Chi2MatchedME0Muon_WideBinning_Eta->Integral()<<std::endl;
  Chi2MuonRecoEff_WideBinning_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_WideBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  
  Chi2MuonRecoEff_WideBinning_Eta->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_WideBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_WideBinning_Eta->SetMinimum(0);
  Chi2MuonRecoEff_WideBinning_Eta->SetMaximum(1.2);

  Chi2MuonRecoEff_WideBinning_Eta->Write();   Chi2MuonRecoEff_WideBinning_Eta->Draw();  

  txt->DrawLatex(0.15,0.2,pcstr);

   
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //latex->SetTextAlign(align_);
  latex->DrawLatex(0.4, 0.85, cmsText);

  c1->Print(histoFolder+"/Chi2MuonRecoEff_WideBinning_Eta.png");

  Chi2MuonRecoEff_WidestBinning_Eta->Divide(Chi2MatchedME0Muon_WidestBinning_Eta, TPMuon_WidestBinning_Eta, 1, 1, "B");
  std::cout<<"TPMuon_WidestBinning_Eta =  "<<TPMuon_WidestBinning_Eta->Integral()<<std::endl;
  std::cout<<"Chi2MatchedME0Muon_WidestBinning_Eta =  "<<Chi2MatchedME0Muon_WidestBinning_Eta->Integral()<<std::endl;
  Chi2MuonRecoEff_WidestBinning_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_WidestBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  
  Chi2MuonRecoEff_WidestBinning_Eta->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_WidestBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_WidestBinning_Eta->SetMinimum(0);
  Chi2MuonRecoEff_WidestBinning_Eta->SetMaximum(1.2);

  Chi2MuonRecoEff_WidestBinning_Eta->Write();   Chi2MuonRecoEff_WidestBinning_Eta->Draw();  

  txt->DrawLatex(0.15,0.2,pcstr);

   
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //latex->SetTextAlign(align_);
  latex->DrawLatex(0.4, 0.85, cmsText);

  c1->Print(histoFolder+"/Chi2MuonRecoEff_WidestBinning_Eta.png");

  std::cout<<"Here0"<<std::endl;

  Chi2MuonRecoEff_Eta_5_10->Divide(Chi2MatchedME0Muon_Eta_5_10, GenMuon_Eta_5_10, 1, 1, "B");
  std::cout<<"Here0"<<std::endl;
  Chi2MuonRecoEff_Eta_5_10->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_Eta_5_10->GetXaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_Eta_5_10->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_Eta_5_10->GetYaxis()->SetTitleSize(0.05);
  //Chi2MuonRecoEff_Eta_5_10->SetMinimum(Chi2MuonRecoEff_Eta_5_10->GetMinimum()-0.1);
  Chi2MuonRecoEff_Eta_5_10->SetMinimum(0);
  //Chi2MuonRecoEff_Eta_5_10->SetMaximum(Chi2MuonRecoEff_Eta_5_10->GetMaximum()+0.1);
  Chi2MuonRecoEff_Eta_5_10->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  Chi2MuonRecoEff_Eta_5_10->Write();   Chi2MuonRecoEff_Eta_5_10->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestChi2MuonRecoEff_Eta_5_10.png");
  c1->Print(histoFolder+"/Chi2MuonRecoEff_Eta_5_10.png");


  Chi2MuonRecoEff_Eta_9_11->Divide(Chi2MatchedME0Muon_Eta_9_11, GenMuon_Eta_9_11, 1, 1, "B");
  std::cout<<"Here0"<<std::endl;
  Chi2MuonRecoEff_Eta_9_11->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_Eta_9_11->GetXaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_Eta_9_11->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_Eta_9_11->GetYaxis()->SetTitleSize(0.05);
  //Chi2MuonRecoEff_Eta_9_11->SetMinimum(Chi2MuonRecoEff_Eta_9_11->GetMinimum()-0.1);
  Chi2MuonRecoEff_Eta_9_11->SetMinimum(0);
  //Chi2MuonRecoEff_Eta_9_11->SetMaximum(Chi2MuonRecoEff_Eta_9_11->GetMaximum()+0.1);
  Chi2MuonRecoEff_Eta_9_11->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  Chi2MuonRecoEff_Eta_9_11->Write();   Chi2MuonRecoEff_Eta_9_11->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestChi2MuonRecoEff_Eta_9_11.png");
  c1->Print(histoFolder+"/Chi2MuonRecoEff_Eta_9_11.png");

  std::cout<<"Here"<<std::endl;

  Chi2MuonRecoEff_Eta_10_50->Divide(Chi2MatchedME0Muon_Eta_10_50, GenMuon_Eta_10_50, 1, 1, "B");
  Chi2MuonRecoEff_Eta_10_50->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_Eta_10_50->GetXaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_Eta_10_50->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_Eta_10_50->GetYaxis()->SetTitleSize(0.05);
  //Chi2MuonRecoEff_Eta_10_50->SetMinimum(Chi2MuonRecoEff_Eta_10_50->GetMinimum()-0.1);
  Chi2MuonRecoEff_Eta_10_50->SetMinimum(0);
  //Chi2MuonRecoEff_Eta_10_50->SetMaximum(Chi2MuonRecoEff_Eta_10_50->GetMaximum()+0.1);
  Chi2MuonRecoEff_Eta_10_50->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  Chi2MuonRecoEff_Eta_10_50->Write();   Chi2MuonRecoEff_Eta_10_50->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestChi2MuonRecoEff_Eta_10_50.png");
  c1->Print(histoFolder+"/Chi2MuonRecoEff_Eta_10_50.png");


  Chi2MuonRecoEff_Eta_50_100->Divide(Chi2MatchedME0Muon_Eta_50_100, GenMuon_Eta_50_100, 1, 1, "B");
  Chi2MuonRecoEff_Eta_50_100->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_Eta_50_100->GetXaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_Eta_50_100->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_Eta_50_100->GetYaxis()->SetTitleSize(0.05);
  //Chi2MuonRecoEff_Eta_50_100->SetMinimum(Chi2MuonRecoEff_Eta_50_100->GetMinimum()-0.1);
  Chi2MuonRecoEff_Eta_50_100->SetMinimum(0);
  //Chi2MuonRecoEff_Eta_50_100->SetMaximum(Chi2MuonRecoEff_Eta_50_100->GetMaximum()+0.1);
  Chi2MuonRecoEff_Eta_50_100->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  Chi2MuonRecoEff_Eta_50_100->Write();   Chi2MuonRecoEff_Eta_50_100->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestChi2MuonRecoEff_Eta_50_100.png");
  c1->Print(histoFolder+"/Chi2MuonRecoEff_Eta_50_100.png");


  Chi2MuonRecoEff_Eta_100->Divide(Chi2MatchedME0Muon_Eta_100, GenMuon_Eta_100, 1, 1, "B");
  Chi2MuonRecoEff_Eta_100->GetXaxis()->SetTitle("Gen Muon |#eta|");
  Chi2MuonRecoEff_Eta_100->GetXaxis()->SetTitleSize(0.05);
  Chi2MuonRecoEff_Eta_100->GetYaxis()->SetTitle("ME0Muon Efficiency");
  Chi2MuonRecoEff_Eta_100->GetYaxis()->SetTitleSize(0.05);
  //Chi2MuonRecoEff_Eta_100->SetMinimum(Chi2MuonRecoEff_Eta_100->GetMinimum()-0.1);
  Chi2MuonRecoEff_Eta_100->SetMinimum(0);
  //Chi2MuonRecoEff_Eta_100->SetMaximum(Chi2MuonRecoEff_Eta_100->GetMaximum()+0.1);
  Chi2MuonRecoEff_Eta_100->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  Chi2MuonRecoEff_Eta_100->Write();   Chi2MuonRecoEff_Eta_100->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestChi2MuonRecoEff_Eta_100.png");
  c1->Print(histoFolder+"/Chi2MuonRecoEff_Eta_100.png");

  std::cout<<"  MuonRecoEff_Eta values:"<<std::endl;
  //MuonRecoEff_Eta->Sumw2();
  for (int i=1; i<=MuonRecoEff_Eta->GetNbinsX(); ++i){
    std::cout<<2.4+(double)i*((4.0-2.4)/40.)<<","<<MuonRecoEff_Eta->GetBinContent(i)<<","<<MuonRecoEff_Eta->GetBinError(i)<<std::endl;
    
  }
  

  // MuonRecoEff_Pt->Divide(MatchedME0Muon_Pt, GenMuon_Pt, 1, 1, "B");
  // MuonRecoEff_Pt->GetXaxis()->SetTitle("Gen Muon p_{T}");
  // MuonRecoEff_Pt->GetYaxis()->SetTitle("Matching Efficiency");
  // MuonRecoEff_Pt->SetMinimum(.85);
  // MuonRecoEff_Pt->Write();   MuonRecoEff_Pt->Draw();  c1->Print(histoFolder+"/MuonRecoEff_Pt.png");

  std::cout<<"UnmatchedME0Muon_Eta =  "<<UnmatchedME0Muon_Cuts_Eta->Integral()<<std::endl;
  std::cout<<"ME0Muon_Eta =  "<<ME0Muon_Cuts_Eta->Integral()<<std::endl;


  FakeRate_Eta->Divide(UnmatchedME0Muon_Cuts_Eta, ME0Muon_Cuts_Eta, 1, 1, "B");
  FakeRate_Eta->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta->SetMinimum(FakeRate_Eta->GetMinimum()-0.1);
  FakeRate_Eta->SetMinimum(0);
  //FakeRate_Eta->SetMaximum(FakeRate_Eta->GetMaximum()+0.1);
  FakeRate_Eta->SetMaximum(1.2);
  FakeRate_Eta->Write();   FakeRate_Eta->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta.png');
  c1->Print(histoFolder+"/FakeRate_Eta.png");

  FakeRate_WideBinning_Eta->Divide(UnmatchedME0Muon_Cuts_WideBinning_Eta, ME0Muon_Cuts_WideBinning_Eta, 1, 1, "B");
  FakeRate_WideBinning_Eta->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_WideBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  FakeRate_WideBinning_Eta->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_WideBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_WideBinning_Eta->SetMinimum(FakeRate_WideBinning_Eta->GetMinimum()-0.1);
  FakeRate_WideBinning_Eta->SetMinimum(0);
  //FakeRate_WideBinning_Eta->SetMaximum(FakeRate_WideBinning_Eta->GetMaximum()+0.1);
  FakeRate_WideBinning_Eta->SetMaximum(1.2);
  FakeRate_WideBinning_Eta->Write();   FakeRate_WideBinning_Eta->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_WideBinning_Eta.png');
  c1->Print(histoFolder+"/FakeRate_WideBinning_Eta.png");


  FakeRate_WidestBinning_Eta->Divide(UnmatchedME0Muon_Cuts_WidestBinning_Eta, ME0Muon_Cuts_WidestBinning_Eta, 1, 1, "B");
  FakeRate_WidestBinning_Eta->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_WidestBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  FakeRate_WidestBinning_Eta->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_WidestBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_WidestBinning_Eta->SetMinimum(FakeRate_WidestBinning_Eta->GetMinimum()-0.1);
  FakeRate_WidestBinning_Eta->SetMinimum(0);
  //FakeRate_WidestBinning_Eta->SetMaximum(FakeRate_WidestBinning_Eta->GetMaximum()+0.1);
  FakeRate_WidestBinning_Eta->SetMaximum(1.2);
  FakeRate_WidestBinning_Eta->Write();   FakeRate_WidestBinning_Eta->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_WidestBinning_Eta.png');
  c1->Print(histoFolder+"/FakeRate_WidestBinning_Eta.png");


  FakeRate_Eta_5_10->Divide(UnmatchedME0Muon_Cuts_Eta_5_10, ME0Muon_Cuts_Eta_5_10, 1, 1, "B");
  FakeRate_Eta_5_10->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_5_10->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_5_10->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_5_10->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_5_10->SetMinimum(FakeRate_Eta_5_10->GetMinimum()-0.1);
  FakeRate_Eta_5_10->SetMinimum(0);
  //FakeRate_Eta_5_10->SetMaximum(FakeRate_Eta_5_10->GetMaximum()+0.1);
  FakeRate_Eta_5_10->SetMaximum(1.2);
  FakeRate_Eta_5_10->Write();   FakeRate_Eta_5_10->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_5_10.png');
  c1->Print(histoFolder+"/FakeRate_Eta_5_10.png");


  FakeRate_Eta_9_11->Divide(UnmatchedME0Muon_Cuts_Eta_9_11, ME0Muon_Cuts_Eta_9_11, 1, 1, "B");
  FakeRate_Eta_9_11->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_9_11->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_9_11->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_9_11->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_9_11->SetMinimum(FakeRate_Eta_9_11->GetMinimum()-0.1);
  FakeRate_Eta_9_11->SetMinimum(0);
  //FakeRate_Eta_9_11->SetMaximum(FakeRate_Eta_9_11->GetMaximum()+0.1);
  FakeRate_Eta_9_11->SetMaximum(1.2);
  FakeRate_Eta_9_11->Write();   FakeRate_Eta_9_11->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_9_11.png');
  c1->Print(histoFolder+"/FakeRate_Eta_9_11.png");



  FakeRate_Eta_10_50->Divide(UnmatchedME0Muon_Cuts_Eta_10_50, ME0Muon_Cuts_Eta_10_50, 1, 1, "B");
  FakeRate_Eta_10_50->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_10_50->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_10_50->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_10_50->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_10_50->SetMinimum(FakeRate_Eta_10_50->GetMinimum()-0.1);
  FakeRate_Eta_10_50->SetMinimum(0);
  //FakeRate_Eta_10_50->SetMaximum(FakeRate_Eta_10_50->GetMaximum()+0.1);
  FakeRate_Eta_10_50->SetMaximum(1.2);
  FakeRate_Eta_10_50->Write();   FakeRate_Eta_10_50->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_10_50.png');
  c1->Print(histoFolder+"/FakeRate_Eta_10_50.png");



  FakeRate_Eta_50_100->Divide(UnmatchedME0Muon_Cuts_Eta_50_100, ME0Muon_Cuts_Eta_50_100, 1, 1, "B");
  FakeRate_Eta_50_100->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_50_100->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_50_100->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_50_100->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_50_100->SetMinimum(FakeRate_Eta_50_100->GetMinimum()-0.1);
  FakeRate_Eta_50_100->SetMinimum(0);
  //FakeRate_Eta_50_100->SetMaximum(FakeRate_Eta_50_100->GetMaximum()+0.1);
  FakeRate_Eta_50_100->SetMaximum(1.2);
  FakeRate_Eta_50_100->Write();   FakeRate_Eta_50_100->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_50_100.png');
  c1->Print(histoFolder+"/FakeRate_Eta_50_100.png");



  FakeRate_Eta_100->Divide(UnmatchedME0Muon_Cuts_Eta_100, ME0Muon_Cuts_Eta_100, 1, 1, "B");
  FakeRate_Eta_100->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_100->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_100->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_100->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_100->SetMinimum(FakeRate_Eta_100->GetMinimum()-0.1);
  FakeRate_Eta_100->SetMinimum(0);
  //FakeRate_Eta_100->SetMaximum(FakeRate_Eta_100->GetMaximum()+0.1);
  FakeRate_Eta_100->SetMaximum(1.2);
  FakeRate_Eta_100->Write();   FakeRate_Eta_100->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_100.png');
  c1->Print(histoFolder+"/FakeRate_Eta_100.png");



  Chi2FakeRate_Eta->Divide(Chi2UnmatchedME0Muon_Eta, ME0Muon_Cuts_Eta, 1, 1, "B");
  std::cout<<"Chi2UnmatchedME0Muon_Eta =  "<<Chi2UnmatchedME0Muon_Eta->Integral()<<std::endl;
  std::cout<<"UnmatchedME0Muon_Eta =  "<<UnmatchedME0Muon_Eta->Integral()<<std::endl;
  std::cout<<"ME0Muon_Eta =  "<<ME0Muon_Cuts_Eta->Integral()<<std::endl;
  std::cout<<"ME0Muon_Eta without cuts =  "<<ME0Muon_Eta->Integral()<<std::endl;

  Chi2FakeRate_Eta->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  Chi2FakeRate_Eta->GetXaxis()->SetTitleSize(0.05);
  Chi2FakeRate_Eta->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  Chi2FakeRate_Eta->GetYaxis()->SetTitleSize(0.05);
  Chi2FakeRate_Eta->SetMinimum(0);
  Chi2FakeRate_Eta->SetMaximum(1.2);
  Chi2FakeRate_Eta->Write();   Chi2FakeRate_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestChi2FakeRate_Eta.png');
  c1->Print(histoFolder+"/Chi2FakeRate_Eta.png");



  Chi2FakeRate_WideBinning_Eta->Divide(Chi2UnmatchedME0Muon_WideBinning_Eta, ME0Muon_Cuts_WideBinning_Eta, 1, 1, "B");
  //std::cout<<"Chi2UnmatchedME0Muon_WideBinning_Eta =  "<<Chi2UnmatchedME0Muon_WideBinning_Eta->Integral()<<std::endl;
  //std::cout<<"UnmatchedME0Muon_WideBinning_Eta =  "<<UnmatchedME0Muon_WideBinning_Eta->Integral()<<std::endl;
  //  std::cout<<"ME0Muon_WideBinning_Eta =  "<<ME0Muon_Cuts_WideBinning_Eta->Integral()<<std::endl;
  //std::cout<<"ME0Muon_WideBinning_Eta without cuts =  "<<ME0Muon_WideBinning_Eta->Integral()<<std::endl;

  Chi2FakeRate_WideBinning_Eta->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  Chi2FakeRate_WideBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  Chi2FakeRate_WideBinning_Eta->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  Chi2FakeRate_WideBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  Chi2FakeRate_WideBinning_Eta->SetMinimum(0);
  Chi2FakeRate_WideBinning_Eta->SetMaximum(1.2);
  Chi2FakeRate_WideBinning_Eta->Write();   Chi2FakeRate_WideBinning_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestChi2FakeRate_WideBinning_Eta.png');
  c1->Print(histoFolder+"/Chi2FakeRate_WideBinning_Eta.png");


  Chi2FakeRate_WidestBinning_Eta->Divide(Chi2UnmatchedME0Muon_WidestBinning_Eta, ME0Muon_Cuts_WidestBinning_Eta, 1, 1, "B");
  //std::cout<<"Chi2UnmatchedME0Muon_WidestBinning_Eta =  "<<Chi2UnmatchedME0Muon_WidestBinning_Eta->Integral()<<std::endl;
  //std::cout<<"UnmatchedME0Muon_WidestBinning_Eta =  "<<UnmatchedME0Muon_WidestBinning_Eta->Integral()<<std::endl;
  //std::cout<<"ME0Muon_WidestBinning_Eta =  "<<ME0Muon_Cuts_WidestBinning_Eta->Integral()<<std::endl;
  //std::cout<<"ME0Muon_WidestBinning_Eta without cuts =  "<<ME0Muon_WidestBinning_Eta->Integral()<<std::endl;

  Chi2FakeRate_WidestBinning_Eta->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  Chi2FakeRate_WidestBinning_Eta->GetXaxis()->SetTitleSize(0.05);
  Chi2FakeRate_WidestBinning_Eta->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  Chi2FakeRate_WidestBinning_Eta->GetYaxis()->SetTitleSize(0.05);
  Chi2FakeRate_WidestBinning_Eta->SetMinimum(0);
  Chi2FakeRate_WidestBinning_Eta->SetMaximum(1.2);
  Chi2FakeRate_WidestBinning_Eta->Write();   Chi2FakeRate_WidestBinning_Eta->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestChi2FakeRate_WidestBinning_Eta.png');
  c1->Print(histoFolder+"/Chi2FakeRate_WidestBinning_Eta.png");


  //Fake Rate per Event:

  FakeRate_Eta_PerEvent->Divide(UnmatchedME0Muon_Cuts_Eta_PerEvent, ME0Muon_Cuts_Eta_PerEvent, 1, 1, "B");
  std::cout<<"UnmatchedME0Muon_Eta_PerEvent =  "<<UnmatchedME0Muon_Cuts_Eta_PerEvent->Integral()<<std::endl;
  std::cout<<"ME0Muon_Eta_PerEvent =  "<<ME0Muon_Cuts_Eta_PerEvent->Integral()<<std::endl;

  FakeRate_Eta_PerEvent->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_PerEvent->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_PerEvent->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_PerEvent->GetYaxis()->SetTitleSize(0.05);
  FakeRate_Eta_PerEvent->SetMinimum(0);
  FakeRate_Eta_PerEvent->SetMaximum(1.2);
  FakeRate_Eta_PerEvent->Write();   FakeRate_Eta_PerEvent->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_PerEvent.png');
  c1->Print(histoFolder+"/FakeRate_Eta_PerEvent.png");

  std::cout<<"  FakeRate_Eta values:"<<std::endl;
  for (int i=1; i<=FakeRate_Eta->GetNbinsX(); ++i){
    std::cout<<2.4+(double)i*((4.0-2.4)/40.)<<","<<FakeRate_Eta->GetBinContent(i)<<std::endl;
  }
  

  // FakeRate_Pt->Divide(MatchedME0Muon_Pt, GenMuon_Pt, 1, 1, "B");
  // FakeRate_Pt->GetXaxis()->SetTitle("Gen Muon p_{T}");
  // FakeRate_Pt->GetYaxis()->SetTitle("Matching Efficiency");
  // FakeRate_Pt->SetMinimum(.85);
  // FakeRate_Pt->Write();   FakeRate_Pt->Draw();  c1->Print(histoFolder+"/FakeRate_Pt.png");

  MuonAllTracksEff_Eta->Divide(ME0Muon_Eta, Track_Eta, 1, 1, "B");
  MuonAllTracksEff_Eta->Write();   MuonAllTracksEff_Eta->Draw();  c1->Print(histoFolder+"/MuonAllTracksEff_Eta.png");

  // MuonAllTracksEff_Pt->Divide(ME0Muon_Pt, Track_Pt, 1, 1, "B");
  // MuonAllTracksEff_Pt->Write();   MuonAllTracksEff_Pt->Draw();  c1->Print(histoFolder+"/MuonAllTracksEff_Pt.png");

  // MuonUnmatchedTracksEff_Eta->Divide(UnmatchedME0Muon_Eta, ME0Muon_Eta, 1, 1, "B");
  // MuonUnmatchedTracksEff_Eta->Write();   Candidate_Eta->Draw();  c1->Print(histoFolder+"/Candidate_Eta.png");

  // MuonUnmatchedTracksEff_Pt->Divide(UnmatchedME0Muon_Pt, ME0Muon_Pt, 1, 1, "B");
  // MuonUnmatchedTracksEff_Pt->Write();   Candidate_Eta->Draw();  c1->Print(histoFolder+"/Candidate_Eta.png");
  FractionMatched_Eta->Divide(MatchedME0Muon_Eta, ME0Muon_Eta, 1, 1, "B");
  FractionMatched_Eta->GetXaxis()->SetTitle("Gen Muon |#eta|");
  FractionMatched_Eta->GetYaxis()->SetTitle("Matched/All ME0Muons");
  FractionMatched_Eta->Write();   FractionMatched_Eta->Draw();  c1->Print(histoFolder+"/FractionMatched_Eta.png");

  gStyle->SetOptStat(1);
  PtDiff_h->GetXaxis()->SetTitle("(pt track-ptgen)/ptgen");
  PtDiff_h->Write();     PtDiff_h->Draw();  c1->Print(histoFolder+"/PtDiff_h.png");

  QOverPtDiff_h->GetXaxis()->SetTitle("(q/pt track-q/ptgen)/(q/ptgen)");
  QOverPtDiff_h->Write();     QOverPtDiff_h->Draw();  c1->Print(histoFolder+"/QOverPtDiff_h.png");

  gStyle->SetOptStat(0);
  PtDiff_s->SetMarkerStyle(1);
  PtDiff_s->SetMarkerSize(3.0);
  PtDiff_s->GetXaxis()->SetTitle("Gen Muon #eta");
  PtDiff_s->GetYaxis()->SetTitle("|(pt track-ptgen)/ptgen|");
  PtDiff_s->Write();     PtDiff_s->Draw();  c1->Print(histoFolder+"/PtDiff_s.png");
  
  for(int i=1; i<=PtDiff_p->GetNbinsX(); ++i) {
    PtDiff_rms->SetBinContent(i, PtDiff_p->GetBinError(i)); 
    std::cout<<"Pt_rms = "<<PtDiff_p->GetBinError(i)<<std::endl;
  }


  TH1D *test;
  std::cout<<"Total integral is "<<PtDiff_s->Integral()<<std::endl;
    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );      
  for(Int_t i=1; i<=PtDiff_s->GetNbinsX(); ++i) {


    std::stringstream tempstore;
    tempstore<<i;
    const std::string& thistemp = tempstore.str();
    test->Draw();


    PtDiff_s->ProjectionY("test",i,i,"");
    std::cout<<"Bin = "<<PtDiff_s->GetBinContent(i)<<std::endl;
    std::cout<<"Integral = "<<test->Integral()<<std::endl;
    if (test->Integral() < 1.0) continue;
    std::cout<<"Running some gaussian fits"<<std::endl;

    // TF1 *gaus_narrow = new TF1("gaus_narrow","gaus",-.1,.1);
    // test->Fit(gaus_narrow,"R");

    // Double_t n0  = gaus_narrow->GetParameter(0);
    // Double_t n1  = gaus_narrow->GetParameter(1);
    // Double_t n2  = gaus_narrow->GetParameter(2);

    // //Double_t e_n0  = gaus_narrow->GetParameterError(0);
    // //Double_t e_n1  = gaus_narrow->GetParameterError(1);
    // Double_t e_n2  = gaus_narrow->GetParError(2);

    // std::cout<<n0<<", "<<n1<<", "<<n2<<std::endl;

    //TF1 *gaus_wide = new TF1("gaus_wide","gaus",-.2,.2);
    TF1 *gaus_wide = new TF1("gaus_wide","gaus",-1.,1.);
    std::cout<<"About to fit"<<std::endl;
    test->Fit(gaus_wide,"R");

    std::cout<<"Getting values"<<std::endl;
    Double_t w2  = gaus_wide->GetParameter(2);

    Double_t e_w2  = gaus_wide->GetParError(2);

    std::cout<<"Got values"<<std::endl;
    // PtDiff_gaus_narrow->SetBinContent(i, n2); 
    // PtDiff_gaus_narrow->SetBinError(i, e_n2); 
    PtDiff_gaus_wide->SetBinContent(i, w2); 
    PtDiff_gaus_wide->SetBinError(i, e_w2); 
    
    test->Write();
    TString FileName = "Bin"+thistemp+"Fit.png";
    c1->Print(histoFolder+"/"+FileName);

    //test->Draw();
    //delete test;
    
    //continue;


    delete test;
    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );  
    test->Draw();
    // Redoing for pt 5 to 10
    std::cout<<"About to project"<<std::endl;
    PtDiff_s_5_10->ProjectionY("test",i,i,"");
    std::cout<<"About to check, "<<std::endl;
    std::cout<<test->Integral()<<std::endl;
    if (test->Integral() < 1.0) continue;

    std::cout<<"Running the 5-10 fit"<<std::endl;
    TF1 *gaus_5_10 = new TF1("gaus_5_10","gaus",-.2,.2);
    test->Fit(gaus_5_10,"R");

     w2  = gaus_5_10->GetParameter(2);
     e_w2  = gaus_5_10->GetParError(2);

    PtDiff_gaus_5_10->SetBinContent(i, w2); 
    PtDiff_gaus_5_10->SetBinError(i, e_w2); 

    test->Draw();
    FileName = "Bin"+thistemp+"Fit_5_10.png";
    c1->Print(histoFolder+"/"+FileName);

    delete test;
    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );  
    test->Draw();
    // Redoing for pt 10 to 20
    PtDiff_s_10_50->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *gaus_10_50 = new TF1("gaus_10_50","gaus",-.2,.2);
    test->Fit(gaus_10_50,"R");

     w2  = gaus_10_50->GetParameter(2);
     e_w2  = gaus_10_50->GetParError(2);

    PtDiff_gaus_10_50->SetBinContent(i, w2); 
    PtDiff_gaus_10_50->SetBinError(i, e_w2); 

    test->Draw();
    FileName = "Bin"+thistemp+"Fit_10_50.png";
    c1->Print(histoFolder+"/"+FileName);

    delete test;

    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );  
    test->Draw();
    // Redoing for pt 20 to 40
    PtDiff_s_50_100->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *gaus_50_100 = new TF1("gaus_50_100","gaus",-.2,.2);
    test->Fit(gaus_50_100,"R");

     w2  = gaus_50_100->GetParameter(2);
     e_w2  = gaus_50_100->GetParError(2);

    PtDiff_gaus_50_100->SetBinContent(i, w2); 
    PtDiff_gaus_50_100->SetBinError(i, e_w2); 

    test->Draw();
    FileName = "Bin"+thistemp+"Fit_50_100.png";
    c1->Print(histoFolder+"/"+FileName);

    delete test;

    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );  
    test->Draw();
    // Redoing for pt 40+
    PtDiff_s_100->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *gaus_100 = new TF1("gaus_100","gaus",-.2,.2);
    test->Fit(gaus_100,"R");

     w2  = gaus_100->GetParameter(2);
     e_w2  = gaus_100->GetParError(2);

    PtDiff_gaus_100->SetBinContent(i, w2); 
    PtDiff_gaus_100->SetBinError(i, e_w2); 

    test->Draw();
    FileName = "Bin"+thistemp+"Fit_100.png";
    c1->Print(histoFolder+"/"+FileName);

    delete test;

    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );  
    test->Draw();
    // Redoing for pt 40+
    StandalonePtDiff_s->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *Standalonegaus = new TF1("Standalonegaus","gaus",-.2,.2);
    test->Fit(Standalonegaus,"R");

     w2  = gaus_100->GetParameter(2);
     e_w2  = gaus_100->GetParError(2);

     StandalonePtDiff_gaus->SetBinContent(i, w2); 
     StandalonePtDiff_gaus->SetBinError(i, e_w2); 

     test->Draw();
     FileName = "Bin"+thistemp+"StandaloneFit.png";
     c1->Print(histoFolder+"/"+FileName);
     
     delete test;
     //test->Clear();
  }
  // PtDiff_gaus_narrow->SetMarkerStyle(22); 
  // PtDiff_gaus_narrow->SetMarkerSize(1.2); 
  // PtDiff_gaus_narrow->SetMarkerColor(kRed); 
  // //PtDiff_gaus_narrow->SetLineColor(kRed); 
  
  // //PtDiff_gaus_narrow->Draw("PL"); 

  // PtDiff_gaus_narrow->GetXaxis()->SetTitle("Gen Muon #eta");
  // PtDiff_gaus_narrow->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  // PtDiff_gaus_narrow->Write();     PtDiff_gaus_narrow->Draw("PE");

  PtDiff_gaus_wide->SetMarkerStyle(22); 
  PtDiff_gaus_wide->SetMarkerSize(1.2); 
  PtDiff_gaus_wide->SetMarkerColor(kBlue); 
  //PtDiff_gaus_wide->SetLineColor(kRed); 
  
  //PtDiff_gaus_wide->Draw("PL"); 

  PtDiff_gaus_wide->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_wide->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_wide->GetYaxis()->SetTitleSize(.04);
  PtDiff_gaus_wide->Write();     PtDiff_gaus_wide->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus.png");

  PtDiff_gaus_5_10->SetMarkerStyle(22); 
  PtDiff_gaus_5_10->SetMarkerSize(1.2); 
  PtDiff_gaus_5_10->SetMarkerColor(kBlue); 
  //PtDiff_gaus_5_10->SetLineColor(kRed); 
  
  //PtDiff_gaus_5_10->Draw("PL"); 

  PtDiff_gaus_5_10->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_5_10->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_5_10->Write();     PtDiff_gaus_5_10->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus_5_10.png");

  PtDiff_gaus_10_50->SetMarkerStyle(22); 
  PtDiff_gaus_10_50->SetMarkerSize(1.2); 
  PtDiff_gaus_10_50->SetMarkerColor(kBlue); 
  //PtDiff_gaus_10_50->SetLineColor(kRed); 
  
  //PtDiff_gaus_10_50->Draw("PL"); 

  PtDiff_gaus_10_50->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_10_50->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_10_50->Write();     PtDiff_gaus_10_50->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus_10_50.png");

  PtDiff_gaus_50_100->SetMarkerStyle(22); 
  PtDiff_gaus_50_100->SetMarkerSize(1.2); 
  PtDiff_gaus_50_100->SetMarkerColor(kBlue); 
  //PtDiff_gaus_50_100->SetLineColor(kRed); 
  
  //PtDiff_gaus_50_100->Draw("PL"); 

  PtDiff_gaus_50_100->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_50_100->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_50_100->Write();     PtDiff_gaus_50_100->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus_50_100.png");

  PtDiff_gaus_100->SetMarkerStyle(22); 
  PtDiff_gaus_100->SetMarkerSize(1.2); 
  PtDiff_gaus_100->SetMarkerColor(kBlue); 
  //PtDiff_gaus_100->SetLineColor(kRed); 
  
  //PtDiff_gaus_100->Draw("PL"); 

  PtDiff_gaus_100->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_100->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_100->Write();     PtDiff_gaus_100->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus_100.png");

  StandalonePtDiff_gaus->SetMarkerStyle(22); 
  StandalonePtDiff_gaus->SetMarkerSize(1.2); 
  StandalonePtDiff_gaus->SetMarkerColor(kBlue); 
  //StandalonePtDiff_gaus->SetLineColor(kRed); 
  
  //StandalonePtDiff_gaus->Draw("PL"); 

  StandalonePtDiff_gaus->GetXaxis()->SetTitle("Gen Muon |#eta|");
  StandalonePtDiff_gaus->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  StandalonePtDiff_gaus->Write();     StandalonePtDiff_gaus->Draw("PE");  c1->Print(histoFolder+"/StandalonePtDiff_gaus.png");


  PtDiff_p->SetMarkerStyle(1);
  PtDiff_p->SetMarkerSize(3.0);
  PtDiff_p->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_p->GetYaxis()->SetTitle("Average (pt track-ptgen)/ptgen");
  PtDiff_p->Write();     PtDiff_p->Draw();  c1->Print(histoFolder+"/PtDiff_p.png");

  //PtDiff_rms->SetMarkerStyle(1);
  //PtDiff_rms->SetMarkerSize(3.0);

  PtDiff_rms->SetMarkerStyle(22); 
  PtDiff_rms->SetMarkerSize(1.2); 
  PtDiff_rms->SetMarkerColor(kBlue); 
  //PtDiff_rms->SetLineColor(kRed); 
  
  //PtDiff_rms->Draw("PL"); 

  PtDiff_rms->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_rms->GetYaxis()->SetTitle("RMS of (pt track-ptgen)/ptgen");
  PtDiff_rms->Write();     PtDiff_rms->Draw("P");  c1->Print(histoFolder+"/PtDiff_rms.png");

  PDiff_h->GetXaxis()->SetTitle("|(p track-pgen)/pgen|");
  PDiff_h->Write();     PDiff_h->Draw();  c1->Print(histoFolder+"/PDiff_h.png");

  PDiff_s->SetMarkerStyle(1);
  PDiff_s->SetMarkerSize(3.0);
  PDiff_s->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PDiff_s->GetYaxis()->SetTitle("|(p track-pgen)/pgen|");
  PDiff_s->Write();     PDiff_s->Draw();  c1->Print(histoFolder+"/PDiff_s.png");
  
  PDiff_p->SetMarkerStyle(1);
  PDiff_p->SetMarkerSize(3.0);
  PDiff_p->GetXaxis()->SetTitle("Gen Muon #eta");
  PDiff_p->GetYaxis()->SetTitle("Average |(p track-pgen)/pgen|");
  PDiff_p->Write();     PDiff_p->Draw();  c1->Print(histoFolder+"/PDiff_p.png");

  VertexDiff_h->Write();     VertexDiff_h->Draw();  c1->Print(histoFolder+"/VertexDiff_h.png");



  NormChi2_h->Write();       NormChi2_h->Draw();          c1->Print(histoFolder+"/NormChi2_h.png");
  NormChi2Prob_h->Write();    NormChi2Prob_h->Draw();      c1->Print(histoFolder+"/NormChi2Prob_h.png");
  NormChi2VsHits_h->Write();    NormChi2VsHits_h->Draw(); c1->Print(histoFolder+"/NormChi2VsHits_h.png");
  chi2_vs_eta_h->Write();      chi2_vs_eta_h->Draw();     c1->Print(histoFolder+"/chi2_vs_eta_h.png");

  c1->SetLogy(0);
  AssociatedChi2_h->Write();    AssociatedChi2_h->Draw();   c1->Print(histoFolder+"/AssociatedChi2_h.png");
  AssociatedChi2_Prob_h->Write();    AssociatedChi2_Prob_h->Draw();   c1->Print(histoFolder+"/AssociatedChi2_Prob_h.png");

  //Printing needing values to an output log file
  using namespace std;
  ofstream logout;
  logout.open (histoFolder+"/Log.txt");

  logout<<"Chi 2 Efficiencies and errors:\n";
  for (int i=1; i<=Chi2MuonRecoEff_Eta->GetNbinsX(); ++i){
    logout<<Chi2MuonRecoEff_Eta->GetBinContent(i)<<","<<Chi2MuonRecoEff_Eta->GetBinError(i)<<"\n";
  }    

  logout<<"Efficiencies and errors:\n";
  for (int i=1; i<=MuonRecoEff_Eta->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta->GetBinContent(i)<<","<<MuonRecoEff_Eta->GetBinError(i)<<"\n";
  }    

  logout<<"Fake Rate:\n";
  for (int i=1; i<=FakeRate_Eta->GetNbinsX(); ++i){
    logout<<FakeRate_Eta->GetBinContent(i)<<","<<FakeRate_Eta->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta:\n";
  for (int i=1; i<=PtDiff_gaus_wide->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_wide->GetBinContent(i)<<","<<PtDiff_gaus_wide->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 5_10:\n";
  for (int i=1; i<=MuonRecoEff_Eta_5_10->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_5_10->GetBinContent(i)<<","<<MuonRecoEff_Eta_5_10->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 9_11:\n";
  for (int i=1; i<=MuonRecoEff_Eta_9_11->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_9_11->GetBinContent(i)<<","<<MuonRecoEff_Eta_9_11->GetBinError(i)<<"\n";
  }    


  logout<<"Chi 2 Efficiencies and errors 5_10:\n";
  for (int i=1; i<=Chi2MuonRecoEff_Eta_5_10->GetNbinsX(); ++i){
    logout<<Chi2MuonRecoEff_Eta_5_10->GetBinContent(i)<<","<<Chi2MuonRecoEff_Eta_5_10->GetBinError(i)<<"\n";
  }    

  logout<<"Fake Rate 5_10:\n";
  for (int i=1; i<=FakeRate_Eta_5_10->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_5_10->GetBinContent(i)<<","<<FakeRate_Eta_5_10->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 5_10:\n";
  for (int i=1; i<=PtDiff_gaus_5_10->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_5_10->GetBinContent(i)<<","<<PtDiff_gaus_5_10->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 10_50:\n";
  for (int i=1; i<=MuonRecoEff_Eta_10_50->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_10_50->GetBinContent(i)<<","<<MuonRecoEff_Eta_10_50->GetBinError(i)<<"\n";
  }    

  logout<<"Chi 2 Efficiencies and errors 10_50:\n";
  for (int i=1; i<=Chi2MuonRecoEff_Eta_10_50->GetNbinsX(); ++i){
    logout<<Chi2MuonRecoEff_Eta_10_50->GetBinContent(i)<<","<<Chi2MuonRecoEff_Eta_10_50->GetBinError(i)<<"\n";
  }    


  logout<<"Fake Rate 10_50:\n";
  for (int i=1; i<=FakeRate_Eta_10_50->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_10_50->GetBinContent(i)<<","<<FakeRate_Eta_10_50->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 10_50:\n";
  for (int i=1; i<=PtDiff_gaus_10_50->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_10_50->GetBinContent(i)<<","<<PtDiff_gaus_10_50->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 50_100:\n";
  for (int i=1; i<=MuonRecoEff_Eta_50_100->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_50_100->GetBinContent(i)<<","<<MuonRecoEff_Eta_50_100->GetBinError(i)<<"\n";
  }    


  logout<<"Chi 2 Efficiencies and errors 50_100:\n";
  for (int i=1; i<=Chi2MuonRecoEff_Eta_50_100->GetNbinsX(); ++i){
    logout<<Chi2MuonRecoEff_Eta_50_100->GetBinContent(i)<<","<<Chi2MuonRecoEff_Eta_50_100->GetBinError(i)<<"\n";
  }    

  logout<<"Fake Rate 50_100:\n";
  for (int i=1; i<=FakeRate_Eta_50_100->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_50_100->GetBinContent(i)<<","<<FakeRate_Eta_50_100->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 50_100:\n";
  for (int i=1; i<=PtDiff_gaus_50_100->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_50_100->GetBinContent(i)<<","<<PtDiff_gaus_50_100->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 40:\n";
  for (int i=1; i<=MuonRecoEff_Eta_100->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_100->GetBinContent(i)<<","<<MuonRecoEff_Eta_100->GetBinError(i)<<"\n";
  }    


  logout<<"Chi 2 Efficiencies and errors 40:\n";
  for (int i=1; i<=Chi2MuonRecoEff_Eta_100->GetNbinsX(); ++i){
    logout<<Chi2MuonRecoEff_Eta_100->GetBinContent(i)<<","<<Chi2MuonRecoEff_Eta_100->GetBinError(i)<<"\n";
  }    

  logout<<"Fake Rate 40:\n";
  for (int i=1; i<=FakeRate_Eta_100->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_100->GetBinContent(i)<<","<<FakeRate_Eta_100->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 40:\n";
  for (int i=1; i<=PtDiff_gaus_100->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_100->GetBinContent(i)<<","<<PtDiff_gaus_100->GetBinError(i)<<"\n";
  }    


  logout<<"Background yield:\n";
  for (int i=1; i<=UnmatchedME0Muon_Cuts_Eta_PerEvent->GetNbinsX(); ++i){
    logout<<UnmatchedME0Muon_Cuts_Eta_PerEvent->GetBinContent(i)<<","<<UnmatchedME0Muon_Cuts_Eta_PerEvent->GetBinError(i)<<"\n";
  }    

  
  //logout << "Writing this to a file.\n";
  logout.close();

  delete histoFile; histoFile = 0;
}



FreeTrajectoryState
ME0MuonAnalyzer::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix55& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CurvilinearTrajectoryError tCov(cov);
  
  return cov.kRows == 5 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

FreeTrajectoryState
ME0MuonAnalyzer::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix66& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);
  
  return cov.kRows == 6 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

void ME0MuonAnalyzer::getFromFTS(const FreeTrajectoryState& fts,
				    GlobalVector& p3, GlobalVector& r3, 
				    int& charge, AlgebraicSymMatrix66& cov){
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  GlobalVector p3T(p3GV.x(), p3GV.y(), p3GV.z());
  GlobalVector r3T(r3GP.x(), r3GP.y(), r3GP.z());
  p3 = p3T;
  r3 = r3T;  //Yikes, was setting this to p3T instead of r3T!?!
  // p3.set(p3GV.x(), p3GV.y(), p3GV.z());
  // r3.set(r3GP.x(), r3GP.y(), r3GP.z());
  
  charge = fts.charge();
  cov = fts.hasError() ? fts.cartesianError().matrix() : AlgebraicSymMatrix66();

}

DEFINE_FWK_MODULE(ME0MuonAnalyzer);
