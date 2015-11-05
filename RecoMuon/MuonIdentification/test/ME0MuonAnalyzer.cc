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

// #include "CLHEP/Matrix/SymMatrix.h"
// #include "CLHEP/Matrix/Matrix.h"
// #include "CLHEP/Vector/ThreeVector.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
//#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TLorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

//#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
//#include "TRandom3.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "DataFormats/Math/interface/deltaR.h"
//#include "DataFormats/Math/interface/deltaPhi.h"
//#include <deltaR.h>
//#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


//Associator for chi2: Including header files
//#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
//#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"

#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"
//#include "CommonTools/CandAlgos/interface/TrackingParticleCustomSelector.h"

//#include "RecoMuon/MuonIdentification/plugins/ME0MuonSelector.cc"

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
//#include "CMSStyle.C"
//#include "tdrstyle.C"
//#include "lumi.C"
//#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

//#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
//#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
//#include <DataFormats/MuonDetId/interface/ME0DetId.h>


#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TGraph.h"

#include <sstream>    

#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class ME0MuonAnalyzer : public edm::EDAnalyzer {
public:
  //ME0MuonAnalyzer(const edm::ParameterSet&,  edm::ConsumesCollector&);
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
  //virtual void endJob();
  //virtual void beginJob(const edm::EventSetup&);
  //void beginJob();
  void beginRun(edm::Run const&, edm::EventSetup const&);
  void endRun(edm::Run const&, edm::EventSetup const&);


  //For Track Association



  //protected:
  
  private:
  edm::EDGetTokenT <reco::GenParticleCollection> genParticlesToken_;
  edm::EDGetTokenT <reco::TrackCollection> globalMuonsToken_;
  //Associator for chi2: objects
  //edm::InputTag associatormap;
  bool UseAssociators;
  bool RejectEndcapMuons;
  //const TrackAssociatorByChi2* associatorByChi2;
  //const TrackAssociatorByHits* associatorByHits;
  // std::vector<std::string> associators;
  // std::vector<const TrackAssociatorBase*> associator;
  //std::vector<edm::InputTag> label;
  //GenParticleCustomSelector gpSelector;	
  //TrackingParticleCustomSelector gpSelector;	
  //std::string parametersDefiner;


  TString histoFolder;
  TString me0MuonSelector;
  TFile* histoFile; 
  // TH1F *Candidate_Eta;  TH1F *Mass_h; 
  // TH1F *Segment_Eta;    TH1F *Segment_Phi;    TH1F *Segment_R;  TH2F *Segment_Pos;  
  // TH1F *Rechit_Eta;    TH1F *Rechit_Phi;    TH1F *Rechit_R;  TH2F *Rechit_Pos;  
  // TH1F *GenMuon_Phi;    TH1F *GenMuon_R;  TH2F *GenMuon_Pos;  
  // TH1F *Track_Eta; TH1F *Track_Pt;  TH1F *ME0Muon_Eta; TH1F *ME0Muon_Pt;  TH1F *CheckME0Muon_Eta; 
  // TH1F *ME0Muon_Cuts_Eta_5_10;   TH1F *ME0Muon_Cuts_Eta_9_11; TH1F *ME0Muon_Cuts_Eta_10_50; TH1F *ME0Muon_Cuts_Eta_50_100; TH1F *ME0Muon_Cuts_Eta_100; 
  // TH1F *UnmatchedME0Muon_Eta; TH1F *UnmatchedME0Muon_Pt;    TH1F *UnmatchedME0Muon_Window_Pt;    TH1F *Chi2UnmatchedME0Muon_Eta; 
  // TH1F *UnmatchedME0Muon_Cuts_Eta_5_10;    TH1F *UnmatchedME0Muon_Cuts_Eta_9_11;  TH1F *UnmatchedME0Muon_Cuts_Eta_10_50;  TH1F *UnmatchedME0Muon_Cuts_Eta_50_100;  TH1F *UnmatchedME0Muon_Cuts_Eta_100;
  // TH1F *TracksPerSegment_h;  TH2F *TracksPerSegment_s;  TProfile *TracksPerSegment_p;
  // TH2F *ClosestDelR_s; TProfile *ClosestDelR_p;
  // TH2F *PtDiff_s; TProfile *PtDiff_p; TH1F *PtDiff_h; TH1F *QOverPtDiff_h; TH1F *PtDiff_rms; TH1F *PtDiff_gaus_narrow; TH1F *PtDiff_gaus_wide;
   TH2F *StandalonePtDiff_s; TProfile *StandalonePtDiff_p; TH1F *StandalonePtDiff_h; TH1F *StandaloneQOverPtDiff_h; TH1F *StandalonePtDiff_rms; TH1F *StandalonePtDiff_gaus_narrow; TH1F *StandalonePtDiff_gaus_wide;
  // TH1F *PtDiff_gaus_5_10;  TH1F *PtDiff_gaus_10_50;  TH1F *PtDiff_gaus_50_100; TH1F *PtDiff_gaus_100;
  TH1F *StandalonePtDiff_gaus;
  // TH1F *VertexDiff_h;
  // TH2F *PDiff_s; TProfile *PDiff_p; TH1F *PDiff_h;
  // TH2F *PtDiff_s_5_10;    TH2F *PtDiff_s_10_50;    TH2F *PtDiff_s_50_100;    TH2F *PtDiff_s_100;
  // TH1F *FakeTracksPerSegment_h;  TH2F *FakeTracksPerSegment_s;  TProfile *FakeTracksPerSegment_p;
  // TH1F *FakeTracksPerAssociatedSegment_h;  TH2F *FakeTracksPerAssociatedSegment_s;  TProfile *FakeTracksPerAssociatedSegment_p;
  TH1F *GenMuon_Eta; TH1F *GenMuon_Pt;
  //TH1F *MatchedME0Muon_Eta; TH1F *MatchedME0Muon_Pt; TH1F *Chi2MatchedME0Muon_Eta; TH1F *Chi2MatchedME0Muon_Pt; 
  TH1F *GenMuon_VariableBins_Pt;
  // TH1F *TPMuon_Eta;
  // TH1F *MatchedME0Muon_Eta_5_10;    TH1F *MatchedME0Muon_Eta_9_11;  TH1F *MatchedME0Muon_Eta_10_50;  TH1F *MatchedME0Muon_Eta_50_100;  TH1F *MatchedME0Muon_Eta_100;
  // TH1F *Chi2MatchedME0Muon_Eta_5_10;   TH1F *Chi2MatchedME0Muon_Eta_9_11; TH1F *Chi2MatchedME0Muon_Eta_10_50;  TH1F *Chi2MatchedME0Muon_Eta_50_100;  TH1F *Chi2MatchedME0Muon_Eta_100;
  // TH1F *GenMuon_Eta_5_10;   TH1F *GenMuon_Eta_9_11;  TH1F *GenMuon_Eta_10_50;  TH1F *GenMuon_Eta_50_100;  TH1F *GenMuon_Eta_100;
  // TH1F *MuonRecoEff_Eta;  TH1F *MuonRecoEff_Pt;   TH1F *Chi2MuonRecoEff_Eta;  
  // TH1F *MuonRecoEff_Eta_5_10;   TH1F *MuonRecoEff_Eta_9_11;  TH1F *MuonRecoEff_Eta_10_50;  TH1F *MuonRecoEff_Eta_50_100;  TH1F *MuonRecoEff_Eta_100;
  // TH1F *Chi2MuonRecoEff_Eta_5_10;    TH1F *Chi2MuonRecoEff_Eta_9_11;  TH1F *Chi2MuonRecoEff_Eta_10_50;  TH1F *Chi2MuonRecoEff_Eta_50_100;  TH1F *Chi2MuonRecoEff_Eta_100;
  // TH1F *FakeRate_Eta;  TH1F *FakeRate_Pt;  TH1F *FakeRate_Eta_PerEvent;    TH1F *Chi2FakeRate_Eta;  

  // TH1F *Chi2FakeRate_WideBinning_Eta;  
  // TH1F *Chi2FakeRate_WidestBinning_Eta;  
  // TH1F *FakeRate_WideBinning_Eta;
  // TH1F *FakeRate_WidestBinning_Eta;
  // TH1F *UnmatchedME0Muon_Cuts_WideBinning_Eta;
  // TH1F *UnmatchedME0Muon_Cuts_WidestBinning_Eta;
  // TH1F *ME0Muon_Cuts_WideBinning_Eta; 
  // TH1F *ME0Muon_Cuts_WidestBinning_Eta;
  // TH1F *Chi2UnmatchedME0Muon_WideBinning_Eta; 
  // TH1F *Chi2UnmatchedME0Muon_WidestBinning_Eta; 
  // TH1F *TPMuon_WideBinning_Eta;
  // TH1F *TPMuon_WidestBinning_Eta;
  TH1F *GenMuon_WideBinning_Eta;
  TH1F *GenMuon_WidestBinning_Eta;
  // TH1F *MatchedME0Muon_WideBinning_Eta;
  // TH1F *MatchedME0Muon_WidestBinning_Eta;
  // TH1F *Chi2MatchedME0Muon_WideBinning_Eta;
  // TH1F *Chi2MatchedME0Muon_WidestBinning_Eta;
  // TH1F *MuonRecoEff_WideBinning_Eta;
  // TH1F *MuonRecoEff_WidestBinning_Eta;
  // TH1F *Chi2MuonRecoEff_WideBinning_Eta;  
  // TH1F *Chi2MuonRecoEff_WidestBinning_Eta;  


  // TH1F *FakeRate_Eta_5_10;    TH1F *FakeRate_Eta_9_11;  TH1F *FakeRate_Eta_10_50;  TH1F *FakeRate_Eta_50_100;  TH1F *FakeRate_Eta_100;
  // TH1F *MuonAllTracksEff_Eta;  TH1F *MuonAllTracksEff_Pt;
  // TH1F *MuonUnmatchedTracksEff_Eta;  TH1F *MuonUnmatchedTracksEff_Pt; TH1F *FractionMatched_Eta;

  TH1F *StandaloneMuonRecoEff_Eta;   TH1F *StandaloneMuonRecoEff_WideBinning_Eta;   TH1F *StandaloneMuonRecoEff_WidestBinning_Eta;
  TH1F *StandaloneMatchedME0Muon_VariableBins_Pt;
  //TH1F *UnmatchedME0Muon_Cuts_Eta;TH1F *ME0Muon_Cuts_Eta;
  TH1F *StandaloneMatchedME0Muon_Eta;    TH1F *StandaloneMatchedME0Muon_WideBinning_Eta;    TH1F *StandaloneMatchedME0Muon_WidestBinning_Eta;
  // TH1F *DelR_Segment_GenMuon;

  // TH1F *SegPosDirPhiDiff_True_h;    TH1F *SegPosDirEtaDiff_True_h;     TH1F *SegPosDirPhiDiff_All_h;    TH1F *SegPosDirEtaDiff_All_h;   
  // TH1F *SegTrackDirPhiDiff_True_h;    TH1F *SegTrackDirEtaDiff_True_h;     TH1F *SegTrackDirPhiDiff_All_h;    TH1F *SegTrackDirEtaDiff_All_h;   TH1F *SegTrackDirPhiPull_True_h;   TH1F *SegTrackDirPhiPull_All_h;   

  // TH1F *SegGenDirPhiDiff_True_h;    TH1F *SegGenDirEtaDiff_True_h;     TH1F *SegGenDirPhiDiff_All_h;    TH1F *SegGenDirEtaDiff_All_h;   TH1F *SegGenDirPhiPull_True_h;   TH1F *SegGenDirPhiPull_All_h;   

  // TH1F *XDiff_h;   TH1F *YDiff_h;   TH1F *XPull_h;   TH1F *YPull_h;


  // TH1F *DelR_Window_Under5; TH1F  *Pt_Window_Under5;
  // TH1F *DelR_Track_Window_Under5; TH1F  *Pt_Track_Window_Under5;  TH1F  *Pt_Track_Window;
  // TH1F *DelR_Track_Window_Failed_Under5; TH1F  *Pt_Track_Window_Failed_Under5;  TH1F  *Pt_Track_Window_Failed;

  // TH1F *FailedTrack_Window_XPull;    TH1F *FailedTrack_Window_YPull;    TH1F *FailedTrack_Window_PhiDiff;
  // TH1F *FailedTrack_Window_XDiff;    TH1F *FailedTrack_Window_YDiff;    

  // TH1F *NormChi2_h;    TH1F *NormChi2Prob_h; TH2F *NormChi2VsHits_h;	TH2F *chi2_vs_eta_h;  TH1F *AssociatedChi2_h;  TH1F *AssociatedChi2_Prob_h;

  // TH1F *PreMatch_TP_R;   TH1F *PostMatch_TP_R;  TH1F *PostMatch_BX0_TP_R;

  double  FakeRatePtCut, MatchingWindowDelR;

  double Nevents;

  
//Removing this
};

ME0MuonAnalyzer::ME0MuonAnalyzer(const edm::ParameterSet& iConfig) 
{
  std::cout<<"Contructor"<<std::endl;
  histoFile = new TFile(iConfig.getParameter<std::string>("HistoFile").c_str(), "recreate");
  histoFolder = iConfig.getParameter<std::string>("HistoFolder").c_str();
  me0MuonSelector = iConfig.getParameter<std::string>("ME0MuonSelectionType").c_str();
  RejectEndcapMuons = iConfig.getParameter< bool >("RejectEndcapMuons");
  UseAssociators = iConfig.getParameter< bool >("UseAssociators");

  FakeRatePtCut   = iConfig.getParameter<double>("FakeRatePtCut");
  MatchingWindowDelR   = iConfig.getParameter<double>("MatchingWindowDelR");

  //Associator for chi2: getting parametters
  //associatormap = iConfig.getParameter< edm::InputTag >("associatormap");
  //UseAssociators = iConfig.getParameter< bool >("UseAssociators");
  //associators = iConfig.getParameter< std::vector<std::string> >("associators");

  //label = iConfig.getParameter< std::vector<edm::InputTag> >("label");

  // gpSelector = GenParticleCustomSelector(iConfig.getParameter<double>("ptMinGP"),
  // 					 iConfig.getParameter<double>("minRapidityGP"),
  // 					 iConfig.getParameter<double>("maxRapidityGP"),
  // 					 iConfig.getParameter<double>("tipGP"),
  // 					 iConfig.getParameter<double>("lipGP"),
  // 					 iConfig.getParameter<bool>("chargedOnlyGP"),
  // 					 iConfig.getParameter<int>("statusGP"),
  // 					 iConfig.getParameter<std::vector<int> >("pdgIdGP"));

    // gpSelector = TrackingParticleCustomSelector(iConfig.getParameter<double>("ptMinGP"),
  // 					 iConfig.getParameter<double>("minRapidityGP"),
  // 					 iConfig.getParameter<double>("maxRapidityGP"),
  // 					 iConfig.getParameter<double>("tipGP"),
  // 					 iConfig.getParameter<double>("lipGP"),
  // 					 iConfig.getParameter<bool>("chargedOnlyGP"),
  // 					 iConfig.getParameter<int>("statusGP"),
  // 					 iConfig.getParameter<std::vector<int> >("pdgIdGP"));
  // //
    //parametersDefiner =iConfig.getParameter<std::string>("parametersDefiner");

  edm::InputTag genParticlesInputTag_("genParticles");
  edm::InputTag globalMuonsInputTag_("globalMuons");
  genParticlesToken_ = consumes<reco::GenParticleCollection>(genParticlesInputTag_);
  globalMuonsToken_  = consumes<reco::TrackCollection>(globalMuonsInputTag_);
  std::cout<<"Contructor end"<<std::endl;
}



//void ME0MuonAnalyzer::beginJob(const edm::EventSetup& iSetup)
void ME0MuonAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {

//void ME0MuonAnalyzer::beginJob()
//{

  std::cout<<"At start of begin run"<<std::endl;  
  //double varbins[]={0.,5.,10.,20.,30.,40.,55.,70.,100.};
  double varbins[]={0.,2.,4.,6.,8.,10.,12.5,15.,20.,30.,40.,55.,70.,100.};
  mkdir(histoFolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  // Candidate_Eta = new TH1F("Candidate_Eta"      , "Candidate #eta"   , 4, 2.0, 2.4 );

  // Track_Eta = new TH1F("Track_Eta"      , "Track #eta"   , 4, 2.0, 2.4 );
  // Track_Pt = new TH1F("Track_Pt"      , "Muon p_{T}"   , 120,0 , 120. );

  // Segment_Eta = new TH1F("Segment_Eta"      , "Segment #eta"   , 4, 2.0, 2.4 );
  // Segment_Phi = new TH1F("Segment_Phi"      , "Segment #phi"   , 60, -3, 3. );
  // Segment_R = new TH1F("Segment_R"      , "Segment r"   , 30, 0, 150 );
  // Segment_Pos = new TH2F("Segment_Pos"      , "Segment x,y"   ,100,-100.,100., 100,-100.,100. );

  // Rechit_Eta = new TH1F("Rechit_Eta"      , "Rechit #eta"   , 4, 2.0, 2.4 );
  // Rechit_Phi = new TH1F("Rechit_Phi"      , "Rechit #phi"   , 60, -3, 3. );
  // Rechit_R = new TH1F("Rechit_R"      , "Rechit r"   , 30, 0, 150 );
  // Rechit_Pos = new TH2F("Rechit_Pos"      , "Rechit x,y"   ,100,-100.,100., 100,-100.,100. );

  // //  GenMuon_Eta = new TH1F("GenMuon_Eta"      , "GenMuon #eta"   , 4, 2.0, 2.4 );
  // GenMuon_Phi = new TH1F("GenMuon_Phi"      , "GenMuon #phi"   , 60, -3, 3. );
  // GenMuon_R = new TH1F("GenMuon_R"      , "GenMuon r"   , 30, 0, 150 );
  // GenMuon_Pos = new TH2F("GenMuon_Pos"      , "GenMuon x,y"   ,100,-100.,100., 100,-100.,100. );

  // ME0Muon_Eta = new TH1F("ME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // ME0Muon_Cuts_Eta_5_10 = new TH1F("ME0Muon_Cuts_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // ME0Muon_Cuts_Eta_9_11 = new TH1F("ME0Muon_Cuts_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // ME0Muon_Cuts_Eta_10_50 = new TH1F("ME0Muon_Cuts_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // ME0Muon_Cuts_Eta_50_100 = new TH1F("ME0Muon_Cuts_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // ME0Muon_Cuts_Eta_100 = new TH1F("ME0Muon_Cuts_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.4 );

  // CheckME0Muon_Eta = new TH1F("CheckME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // ME0Muon_Pt = new TH1F("ME0Muon_Pt"      , "Muon p_{T}"   , 120,0 , 120. );

  GenMuon_Eta = new TH1F("GenMuon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // GenMuon_Eta_5_10 = new TH1F("GenMuon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // GenMuon_Eta_9_11 = new TH1F("GenMuon_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // GenMuon_Eta_10_50 = new TH1F("GenMuon_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // GenMuon_Eta_50_100 = new TH1F("GenMuon_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // GenMuon_Eta_100 = new TH1F("GenMuon_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.4 );

  // TPMuon_Eta = new TH1F("TPMuon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );

  // GenMuon_Pt = new TH1F("GenMuon_Pt"      , "Muon p_{T}"   , 120,0 , 120. );
  GenMuon_VariableBins_Pt = new TH1F("GenMuon_VariableBins_Pt"      , "Muon p_{T}"   ,13,varbins);

  //MatchedME0Muon_Eta = new TH1F("MatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  StandaloneMatchedME0Muon_Eta = new TH1F("StandaloneMatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  StandaloneMatchedME0Muon_WideBinning_Eta = new TH1F("StandaloneMatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 12, 0., 2.4 );
  StandaloneMatchedME0Muon_WidestBinning_Eta = new TH1F("StandaloneMatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 24, 0., 2.4 );
  // StandaloneMatchedME0Muon_WideBinning_Eta = new TH1F("StandaloneMatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // StandaloneMatchedME0Muon_WidestBinning_Eta = new TH1F("StandaloneMatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );
  // MatchedME0Muon_Eta_5_10 = new TH1F("MatchedME0Muon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // MatchedME0Muon_Eta_9_11 = new TH1F("MatchedME0Muon_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // MatchedME0Muon_Eta_10_50 = new TH1F("MatchedME0Muon_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // MatchedME0Muon_Eta_50_100 = new TH1F("MatchedME0Muon_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // MatchedME0Muon_Eta_100 = new TH1F("MatchedME0Muon_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.4 );


  // Chi2MatchedME0Muon_Eta_5_10 = new TH1F("Chi2MatchedME0Muon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // Chi2MatchedME0Muon_Eta_9_11 = new TH1F("Chi2MatchedME0Muon_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // Chi2MatchedME0Muon_Eta_10_50 = new TH1F("Chi2MatchedME0Muon_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // Chi2MatchedME0Muon_Eta_50_100 = new TH1F("Chi2MatchedME0Muon_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // Chi2MatchedME0Muon_Eta_100 = new TH1F("Chi2MatchedME0Muon_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.4 );

  // MatchedME0Muon_Pt = new TH1F("MatchedME0Muon_Pt"      , "Muon p_{T}"   , 40,0 , 20 );

  // Chi2MatchedME0Muon_Eta = new TH1F("Chi2MatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // Chi2MatchedME0Muon_Pt = new TH1F("Chi2MatchedME0Muon_Pt"      , "Muon p_{T}"   , 40,0 , 20 );

  // Chi2UnmatchedME0Muon_Eta = new TH1F("Chi2UnmatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );

  // UnmatchedME0Muon_Eta = new TH1F("UnmatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // UnmatchedME0Muon_Cuts_Eta_5_10 = new TH1F("UnmatchedME0Muon_Cuts_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // UnmatchedME0Muon_Cuts_Eta_9_11 = new TH1F("UnmatchedME0Muon_Cuts_Eta_9_11"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // UnmatchedME0Muon_Cuts_Eta_10_50 = new TH1F("UnmatchedME0Muon_Cuts_Eta_10_50"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // UnmatchedME0Muon_Cuts_Eta_50_100 = new TH1F("UnmatchedME0Muon_Cuts_Eta_50_100"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // UnmatchedME0Muon_Cuts_Eta_100 = new TH1F("UnmatchedME0Muon_Cuts_Eta_100"      , "Muon #eta"   , 4, 2.0, 2.4 );

  //UnmatchedME0Muon_Pt = new TH1F("UnmatchedME0Muon// _Pt"      , "Muon p_{T}"   , 500,0 , 50 );
  // UnmatchedME0Muon_Window_Pt = new TH1F("UnmatchedME0Muon_Window_Pt"      , "Muon p_{T}"   , 500,0 , 50 );

  // UnmatchedME0Muon_Cuts_Eta = new TH1F("UnmatchedME0Muon_Cuts_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );
  // ME0Muon_Cuts_Eta = new TH1F("ME0Muon_Cuts_Eta"      , "Muon #eta"   , 4, 2.0, 2.4 );

  // Mass_h = new TH1F("Mass_h"      , "Mass"   , 100, 0., 200 );

  // MuonRecoEff_Eta = new TH1F("MuonRecoEff_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );

  // MuonRecoEff_Eta_5_10 = new TH1F("MuonRecoEff_Eta_5_10"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // MuonRecoEff_Eta_9_11 = new TH1F("MuonRecoEff_Eta_9_11"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // MuonRecoEff_Eta_10_50 = new TH1F("MuonRecoEff_Eta_10_50"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // MuonRecoEff_Eta_50_100 = new TH1F("MuonRecoEff_Eta_50_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // MuonRecoEff_Eta_100 = new TH1F("MuonRecoEff_Eta_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // Chi2MuonRecoEff_Eta = new TH1F("Chi2MuonRecoEff_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // Chi2MuonRecoEff_Eta_5_10 = new TH1F("Chi2MuonRecoEff_Eta_5_10"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // Chi2MuonRecoEff_Eta_9_11 = new TH1F("Chi2MuonRecoEff_Eta_9_11"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // Chi2MuonRecoEff_Eta_10_50 = new TH1F("Chi2MuonRecoEff_Eta_10_50"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // Chi2MuonRecoEff_Eta_50_100 = new TH1F("Chi2MuonRecoEff_Eta_50_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );
  // Chi2MuonRecoEff_Eta_100 = new TH1F("Chi2MuonRecoEff_Eta_100"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.4  );

  // MuonRecoEff_Pt = new TH1F("MuonRecoEff_Pt"      , "Fraction of ME0Muons matched to gen muons"   ,8, 0,40  );

  StandaloneMuonRecoEff_Eta = new TH1F("StandaloneMuonRecoEff_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,4, 2.0, 2.4  );
  StandaloneMuonRecoEff_WideBinning_Eta = new TH1F("StandaloneMuonRecoEff_WideBinning_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,12, 0., 2.4  );
  StandaloneMuonRecoEff_WidestBinning_Eta = new TH1F("StandaloneMuonRecoEff_WidestBinning_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,24, 0., 2.4  );
  // // StandaloneMuonRecoEff_WideBinning_Eta = new TH1F("StandaloneMuonRecoEff_WideBinning_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,8, 2.0, 2.4  );
  // // StandaloneMuonRecoEff_WidestBinning_Eta = new TH1F("StandaloneMuonRecoEff_WidestBinning_Eta"      , "Fraction of Standalone Muons matched to gen muons"   ,16, 2.0, 2.4  );

  // FakeRate_Eta = new TH1F("FakeRate_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.4  );
  // FakeRate_Eta_5_10 = new TH1F("FakeRate_Eta_5_10"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.4  );
  // FakeRate_Eta_9_11 = new TH1F("FakeRate_Eta_9_11"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.4  );
  // FakeRate_Eta_10_50 = new TH1F("FakeRate_Eta_10_50"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.4  );
  // FakeRate_Eta_50_100 = new TH1F("FakeRate_Eta_50_100"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.4  );
  // FakeRate_Eta_100 = new TH1F("FakeRate_Eta_100"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.4  );

  // Chi2FakeRate_Eta = new TH1F("Chi2FakeRate_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.4  );

  // FakeRate_Eta_PerEvent = new TH1F("FakeRate_Eta_PerEvent"      , "PU140, unmatched ME0Muons/all ME0Muons normalized by N_{events}"   ,4, 2.0, 2.4  );
  // FakeRate_Pt = new TH1F("FakeRate_Pt"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,8, 0,40  );

  // MuonAllTracksEff_Eta = new TH1F("MuonAllTracksEff_Eta"      , "All ME0Muons over all tracks"   ,4, 2.0, 2.4  );
  // MuonAllTracksEff_Pt = new TH1F("MuonAllTracksEff_Pt"      , "All ME0Muons over all tracks"   ,8, 0,40  );

  // MuonUnmatchedTracksEff_Eta = new TH1F("MuonUnmatchedTracksEff_Eta"      , "Unmatched ME0Muons over all ME0Muons"   ,4, 2.0, 2.4  );
  // MuonUnmatchedTracksEff_Pt = new TH1F("MuonUnmatchedTracksEff_Pt"      , "Unmatched ME0Muons over all ME0Muons"   ,8, 0,40  );

  // TracksPerSegment_h = new TH1F("TracksPerSegment_h", "Number of tracks", 60,0.,60.);
  // TracksPerSegment_s = new TH2F("TracksPerSegment_s" , "Tracks per segment vs |#eta|", 4, 2.0, 2.4, 60,0.,60.);
  // TracksPerSegment_p = new TProfile("TracksPerSegment_p" , "Tracks per segment vs |#eta|", 4, 2.0, 2.4, 0.,60.);

  // FakeTracksPerSegment_h = new TH1F("FakeTracksPerSegment_h", "Number of fake tracks", 60,0.,60.);
  // FakeTracksPerSegment_s = new TH2F("FakeTracksPerSegment_s" , "Fake tracks per segment", 10, 2.4, 4.0, 100,0.,60.);
  // FakeTracksPerSegment_p = new TProfile("FakeTracksPerSegment_p" , "Average N_{tracks}/segment not matched to genmuons", 10, 2.4, 4.0, 0.,60.);

  // FakeTracksPerAssociatedSegment_h = new TH1F("FakeTracksPerAssociatedSegment_h", "Number of fake tracks", 60,0.,60.);
  // FakeTracksPerAssociatedSegment_s = new TH2F("FakeTracksPerAssociatedSegment_s" , "Fake tracks per segment", 10, 2.4, 4.0, 100,0.,60.);
  // FakeTracksPerAssociatedSegment_p = new TProfile("FakeTracksPerAssociatedSegment_p" , "Average N_{tracks}/segment not matched to genmuons", 10, 2.4, 4.0, 0.,60.);

  // ClosestDelR_s = new TH2F("ClosestDelR_s" , "#Delta R", 4, 2.0, 2.4, 15,0.,0.15);
  // ClosestDelR_p = new TProfile("ClosestDelR_p" , "#Delta R", 4, 2.0, 2.4, 0.,0.15);

  // DelR_Window_Under5 = new TH1F("DelR_Window_Under5","#Delta R", 15, 0,0.15  );
  // Pt_Window_Under5 = new TH1F("Pt_Window_Under5","pt",500, 0,50  );

  // DelR_Track_Window_Under5 = new TH1F("DelR_Track_Window_Under5","#Delta R", 15, 0,0.15  );
  // Pt_Track_Window_Under5 = new TH1F("Pt_Track_Window_Under5","pt",20, 0,5  );
  // Pt_Track_Window = new TH1F("Pt_Track_Window","pt",500, 0,  50);

  // DelR_Track_Window_Failed_Under5 = new TH1F("DelR_Track_Window_Failed_Under5","#Delta R", 15, 0,0.15  );
  // Pt_Track_Window_Failed_Under5 = new TH1F("Pt_Track_Window_Failed_Under5","pt",20, 0,5  );
  // Pt_Track_Window_Failed = new TH1F("Pt_Track_Window_Failed","pt",500, 0,  50);

  // FailedTrack_Window_XPull = new TH1F("FailedTrack_Window_XPull", "X Pull failed tracks", 100, 0,20);
  // FailedTrack_Window_YPull = new TH1F("FailedTrack_Window_YPull", "Y  Pull failed tracks", 100, 0,20);
  // FailedTrack_Window_XDiff = new TH1F("FailedTrack_Window_XDiff", "X Diff failed tracks", 100, 0,20);
  // FailedTrack_Window_YDiff = new TH1F("FailedTrack_Window_YDiff", "Y  Diff failed tracks", 100, 0,20);

  // FailedTrack_Window_PhiDiff = new TH1F("FailedTrack_Window_PhiDiff", "Phi Dir Diff failed tracks", 100,0 ,2.0);

  // DelR_Segment_GenMuon = new TH1F("DelR_Segment_GenMuon", "#Delta R between me0segment and gen muon",200,0,2);
  // FractionMatched_Eta = new TH1F("FractionMatched_Eta"      , "Fraction of ME0Muons that end up successfully matched (matched/all)"   ,4, 2.0, 2.4  );

  // PtDiff_s = new TH2F("PtDiff_s" , "Relative pt difference", 4, 2.0, 2.4, 200,-1,1.0);

  // PtDiff_s_5_10 = new TH2F("PtDiff_s_5_10" , "Relative pt difference", 4, 2.0, 2.4, 200,-1,1.0);
  // PtDiff_s_10_50 = new TH2F("PtDiff_s_10_50" , "Relative pt difference", 4, 2.0, 2.4, 200,-1,1.0);
  // PtDiff_s_50_100 = new TH2F("PtDiff_s_50_100" , "Relative pt difference", 4, 2.0, 2.4, 200,-1,1.0);
  // PtDiff_s_100 = new TH2F("PtDiff_s_100" , "Relative pt difference", 4, 2.0, 2.4, 200,-1,1.0);

  // PtDiff_h = new TH1F("PtDiff_h" , "pt resolution", 100,-0.5,0.5);
  // QOverPtDiff_h = new TH1F("QOverPtDiff_h" , "q/pt resolution", 100,-0.5,0.5);
  // PtDiff_p = new TProfile("PtDiff_p" , "pt resolution vs. #eta", 4, 2.0, 2.4, -1.0,1.0,"s");

  StandalonePtDiff_s = new TH2F("StandalonePtDiff_s" , "Relative pt difference", 4, 2.0, 2.4, 200,-1,1.0);
  StandalonePtDiff_h = new TH1F("StandalonePtDiff_h" , "pt resolution", 100,-0.5,0.5);
  StandaloneQOverPtDiff_h = new TH1F("StandaloneQOverPtDiff_h" , "q/pt resolution", 100,-0.5,0.5);
  // StandalonePtDiff_p = new TProfile("StandalonePtDiff_p" , "pt resolution vs. #eta", 4, 2.0, 2.4, -1.0,1.0,"s");

  StandaloneMatchedME0Muon_VariableBins_Pt = new TH1F("StandaloneMatchedME0Muon_VariableBins_Pt"      , "Muon p_{T}"   ,13,varbins);

  // PtDiff_rms    = new TH1F( "PtDiff_rms",    "RMS", 4, 2.0, 2.4 ); 
  // PtDiff_gaus_wide    = new TH1F( "PtDiff_gaus_wide",    "GAUS_WIDE", 4, 2.0, 2.4 ); 
  // PtDiff_gaus_narrow    = new TH1F( "PtDiff_gaus_narrow",    "GAUS_NARROW", 4, 2.0, 2.4 ); 

  // PtDiff_gaus_5_10    = new TH1F( "PtDiff_gaus_5_10",    "GAUS_WIDE", 4, 2.0, 2.4 ); 
  // PtDiff_gaus_10_50    = new TH1F( "PtDiff_gaus_10_50",    "GAUS_WIDE", 4, 2.0, 2.4 ); 
  // PtDiff_gaus_50_100    = new TH1F( "PtDiff_gaus_50_100",    "GAUS_WIDE", 4, 2.0, 2.4 ); 
  // PtDiff_gaus_100    = new TH1F( "PtDiff_gaus_100",    "GAUS_WIDE", 4, 2.0, 2.4 ); 

  StandalonePtDiff_gaus    = new TH1F( "StandalonePtDiff_gaus",    "GAUS_WIDE", 4, 2.0, 2.4 ); 

  // PDiff_s = new TH2F("PDiff_s" , "Relative p difference", 4, 2.0, 2.4, 50,0.,0.5);
  // PDiff_h = new TH1F("PDiff_s" , "Relative p difference", 50,0.,0.5);
  // PDiff_p = new TProfile("PDiff_p" , "Relative p difference", 4, 2.0, 2.4, 0.,1.0,"s");

  // VertexDiff_h = new TH1F("VertexDiff_h", "Difference in vertex Z", 50, 0, 0.2);

  // SegPosDirPhiDiff_True_h = new TH1F("SegPosDirPhiDiff_True_h", "#phi Dir. Diff. Real Muons", 50, -2,2);
  // SegPosDirEtaDiff_True_h = new TH1F("SegPosDirEtaDiff_True_h", "#eta Dir. Diff. Real Muons", 50, -2,2);

  // SegPosDirPhiDiff_All_h = new TH1F("SegPosDirPhiDiff_All_h", "#phi Dir. Diff. All Muons", 50, -3,3);
  // SegPosDirEtaDiff_All_h = new TH1F("SegPosDirEtaDiff_All_h", "#eta Dir. Diff. All Muons", 50, -3,3);

  // SegTrackDirPhiDiff_True_h = new TH1F("SegTrackDirPhiDiff_True_h", "#phi Dir. Diff. Real Muons", 50, -2,2);
  // SegTrackDirEtaDiff_True_h = new TH1F("SegTrackDirEtaDiff_True_h", "#eta Dir. Diff. Real Muons", 50, -2,2);

  // SegTrackDirPhiPull_True_h = new TH1F("SegTrackDirPhiPull_True_h", "#phi Dir. Pull. Real Muons", 50, -3,3);
  // SegTrackDirPhiPull_All_h = new TH1F("SegTrackDirPhiPull_True_h", "#phi Dir. Pull. All Muons", 50, -3,3);

  // SegTrackDirPhiDiff_All_h = new TH1F("SegTrackDirPhiDiff_All_h", "#phi Dir. Diff. All Muons", 50, -3,3);
  // SegTrackDirEtaDiff_All_h = new TH1F("SegTrackDirEtaDiff_All_h", "#eta Dir. Diff. All Muons", 50, -3,3);

  // SegGenDirPhiDiff_True_h = new TH1F("SegGenDirPhiDiff_True_h", "#phi Dir. Diff. Real Muons", 50, -2,2);
  // SegGenDirEtaDiff_True_h = new TH1F("SegGenDirEtaDiff_True_h", "#eta Dir. Diff. Real Muons", 50, -2,2);

  // SegGenDirPhiPull_True_h = new TH1F("SegGenDirPhiPull_True_h", "#phi Dir. Pull. Real Muons", 50, -3,3);
  // SegGenDirPhiPull_All_h = new TH1F("SegGenDirPhiPull_True_h", "#phi Dir. Pull. All Muons", 50, -3,3);

  // SegGenDirPhiDiff_All_h = new TH1F("SegGenDirPhiDiff_All_h", "#phi Dir. Diff. All Muons", 50, -3,3);
  // SegGenDirEtaDiff_All_h = new TH1F("SegGenDirEtaDiff_All_h", "#eta Dir. Diff. All Muons", 50, -3,3);


  // PreMatch_TP_R = new TH1F("PreMatch_TP_R", "r distance from TP pre match to beamline", 100, 0, 10);
  // PostMatch_TP_R = new TH1F("PostMatch_TP_R", "r distance from TP post match to beamline", 200, 0, 20);
  // PostMatch_BX0_TP_R = new TH1F("PostMatch_BX0_TP_R", "r distance from TP post match to beamline", 200, 0, 20);


  // XDiff_h = new TH1F("XDiff_h", "X Diff", 100, -10.0, 10.0 );
  // YDiff_h = new TH1F("YDiff_h", "Y Diff", 100, -50.0, 50.0 ); 
  // XPull_h = new TH1F("XPull_h", "X Pull", 100, -5.0, 5.0 );
  // YPull_h = new TH1F("YPull_h", "Y Pull", 40, -50.0, 50.0 );

  // MuonRecoEff_WideBinning_Eta = new TH1F("MuonRecoEff_WideBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,8, 2.0, 2.4  );
  // MuonRecoEff_WidestBinning_Eta = new TH1F("MuonRecoEff_WidestBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,16, 2.0, 2.4  );
  // Chi2MuonRecoEff_WideBinning_Eta = new TH1F("Chi2MuonRecoEff_WideBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,8, 2.0, 2.4  );
  // Chi2MuonRecoEff_WidestBinning_Eta = new TH1F("Chi2MuonRecoEff_WidestBinning_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,16, 2.0, 2.4  );
  // Chi2FakeRate_WideBinning_Eta = new TH1F("Chi2FakeRate_WideBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,8, 2.0, 2.4  );
  // Chi2FakeRate_WidestBinning_Eta = new TH1F("Chi2FakeRate_WidestBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,16, 2.0, 2.4  );
  // FakeRate_WideBinning_Eta = new TH1F("FakeRate_WideBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,8, 2.0, 2.4  );
  // FakeRate_WidestBinning_Eta = new TH1F("FakeRate_WidestBinning_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons// "   ,16, 2.0, 2.4  );

  // UnmatchedME0Muon_Cuts_WideBinning_Eta = new TH1F("UnmatchedME0Muon_Cuts_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // UnmatchedME0Muon_Cuts_WidestBinning_Eta = new TH1F("UnmatchedME0Muon_Cuts_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );
  // ME0Muon_Cuts_WideBinning_Eta = new TH1F("ME0Muon_Cuts_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // ME0Muon_Cuts_WidestBinning_Eta = new TH1F("ME0Muon_Cuts_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );
  // Chi2UnmatchedME0Muon_WideBinning_Eta = new TH1F("Chi2UnmatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // Chi2UnmatchedME0Muon_WidestBinning_Eta = new TH1F("Chi2UnmatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );
  // TPMuon_WideBinning_Eta = new TH1F("TPMuon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // TPMuon_WidestBinning_Eta = new TH1F("TPMuon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );
  GenMuon_WideBinning_Eta = new TH1F("GenMuon_WideBinning_Eta"      , "Muon #eta"   , 12, 0., 2.4 );
  GenMuon_WidestBinning_Eta = new TH1F("GenMuon_WidestBinning_Eta"      , "Muon #eta"   , 24, 0., 2.4 );
  // // GenMuon_WideBinning_Eta = new TH1F("GenMuon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // // GenMuon_WidestBinning_Eta = new TH1F("GenMuon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );
  // MatchedME0Muon_WideBinning_Eta = new TH1F("MatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // MatchedME0Muon_WidestBinning_Eta = new TH1F("MatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );
  // Chi2MatchedME0Muon_WideBinning_Eta = new TH1F("Chi2MatchedME0Muon_WideBinning_Eta"      , "Muon #eta"   , 8, 2.0, 2.4 );
  // Chi2MatchedME0Muon_WidestBinning_Eta = new TH1F("Chi2MatchedME0Muon_WidestBinning_Eta"      , "Muon #eta"   , 16, 2.0, 2.4 );

 
  // AssociatedChi2_h = new TH1F("AssociatedChi2_h","Associated #chi^{2}",50,0,50);
  // AssociatedChi2_Prob_h = new TH1F("AssociatedChi2_h","Associated #chi^{2}",50,0,1);
  // NormChi2_h = new TH1F("NormChi2_h","normalized #chi^{2}", 200, 0, 20);
  // NormChi2Prob_h = new TH1F("NormChi2Prob_h","normalized #chi^{2} probability", 100, 0, 1);
  // NormChi2VsHits_h = new TH2F("NormChi2VsHits_h","#chi^{2} vs nhits",25,0,25,100,0,10);
  // chi2_vs_eta_h = new TH2F("chi2_vs_eta_h","#chi^{2} vs #eta",4, 2.0, 2.4 , 200, 0, 20);

  Nevents=0;
  // std::cout<<"HERE NOW, about to check if get associator"<<std::endl;
  // if (UseAssociators) {
  //   std::cout<<"Getting the associator"<<std::endl;
  //   edm::ESHandle<TrackAssociatorBase> theAssociator;
  //   for (unsigned int w=0;w<associators.size();w++) {
  //     iSetup.get<TrackAssociatorRecord>().get(associators[w],theAssociator);
  //     associator.push_back( theAssociator.product() );
  //   }
  // }
  // std::cout<<"HERE NOW"<<std::endl;

}


ME0MuonAnalyzer::~ME0MuonAnalyzer(){}

void
ME0MuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)

{

  std::cout<<"ANALYZER"<<std::endl;
  
  using namespace edm;


  using namespace reco;

  edm::Handle<GenParticleCollection> genParticles;

  //iEvent.getByLabel<GenParticleCollection>("genParticles", genParticles);
  iEvent.getByToken<GenParticleCollection>(genParticlesToken_, genParticles);

  unsigned int gensize=genParticles->size();




  Nevents++;

  std::cout<<"About to get muons:"<<std::endl;
  //edm::Handle<std::vector<Muon> > muons;
  //iEvent.getByLabel("Muons", muons);
  //iEvent.getByLabel("muons", muons);

  edm::Handle<TrackCollection> muons;
  iEvent.getByToken <TrackCollection> (globalMuonsToken_,muons);
  
  std::cout<<"Have muons, about to start"<<std::endl;
  for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
      std::cout<<"here"<<std::endl;
      double LowestDelR = 9999;
      double thisDelR = 9999;
      
      //std::vector<double> ReferenceTrackPt;
      if (CurrentParticle.pt() > FakeRatePtCut ) {
	GenMuon_Eta->Fill(fabs(CurrentParticle.eta()));
	GenMuon_WideBinning_Eta->Fill(fabs(CurrentParticle.eta()));
	GenMuon_WidestBinning_Eta->Fill(fabs(CurrentParticle.eta()));
	//if ( (fabs(CurrentParticle.eta()) > 2.0) && (fabs(CurrentParticle.eta()) < 2.8) ) {
	if (  (fabs(CurrentParticle.eta()) < 2.4) ) {
	  GenMuon_VariableBins_Pt->Fill(CurrentParticle.pt());
	}
	  //}

      }
      std::cout<<"here"<<std::endl;
	//double VertexDiff=-1,PtDiff=-1,QOverPtDiff=-1,PDiff=-1,MatchedEta=-1;
      double PtDiff=-1,QOverPtDiff=-1,MatchedEta=-1,MatchedPt=-1;

      std::cout<<"Size = "<<muons->size()<<std::endl;
      // for (std::vector<Muon>::const_iterator thisMuon = muons->begin();
      // 	   thisMuon != muons->end(); ++thisMuon){
      for (std::vector<Track>::const_iterator thisMuon = muons->begin();
	   thisMuon != muons->end(); ++thisMuon){
  	
	//TrackRef tkRef = thisMuon->outerTrack();
	thisDelR = reco::deltaR(CurrentParticle,*thisMuon);
	if (CurrentParticle.pt() > FakeRatePtCut ) {
	  if (thisDelR < MatchingWindowDelR ){
	    if (thisDelR < LowestDelR){
	      LowestDelR = thisDelR;
	      //if (thisMuon->outerTrack()->hitPattern().muonStationsWithValidHits() > 1 && abs(thisMuon->time().timeAtIpInOut) < (12.5 + abs(thisMuon->time().timeAtIpInOutErr))){
	      MatchedEta=fabs(CurrentParticle.eta());
	      //VertexDiff = fabs(thisMuon->vz()-CurrentParticle.vz());
	      QOverPtDiff = ( (thisMuon->charge() /thisMuon->pt()) - (CurrentParticle.charge()/CurrentParticle.pt() ) )/  (CurrentParticle.charge()/CurrentParticle.pt() );
	      PtDiff = (thisMuon->pt() - CurrentParticle.pt())/CurrentParticle.pt();
	      //if ( (fabs(CurrentParticle.eta()) > 2.0) && (fabs(CurrentParticle.eta()) < 2.8) ) {
	      if (  (fabs(CurrentParticle.eta()) < 2.4) ) {
		MatchedPt=CurrentParticle.pt();
	      }
	      //}
	      
	      //PDiff = (tkRef->p() - CurrentParticle.p())/CurrentParticle.p();
	      //}
	    }
	  }
	}
      }
      
      StandaloneMatchedME0Muon_Eta->Fill(MatchedEta);
      StandaloneMatchedME0Muon_WideBinning_Eta->Fill(MatchedEta);
      StandaloneMatchedME0Muon_WidestBinning_Eta->Fill(MatchedEta);
      StandaloneMatchedME0Muon_VariableBins_Pt->Fill(MatchedPt);
      //StandaloneVertexDiff_h->Fill(VertexDiff);
      StandalonePtDiff_h->Fill(PtDiff);	
      StandaloneQOverPtDiff_h->Fill(QOverPtDiff);
      StandalonePtDiff_s->Fill(CurrentParticle.eta(),PtDiff);

    }
  }
  
  
}

//void ME0MuonAnalyzer::endJob() 
void ME0MuonAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) 

{

  std::cout<<"Nevents = "<<Nevents<<std::endl;
  //TString cmsText     = "CMS Prelim.";
  //TString cmsText     = "#splitline{CMS PhaseII Simulation}{Prelim}";
  TString cmsText     = "CMS PhaseII Simulation Prelim.";

  TString lumiText = "PU 140, 14 TeV";
  float cmsTextFont   = 61;  // default is helvetic-bold

 
  //float extraTextFont = 52;  // default is helvetica-italics
  //float lumiTextSize     = 0.05;

  //float lumiTextOffset   = 0.2;
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

  GenMuon_Eta->Write();   GenMuon_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_Eta.png");
  GenMuon_WideBinning_Eta->Write();   GenMuon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_WideBinning_Eta.png");
  GenMuon_WidestBinning_Eta->Write();   GenMuon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_WidestBinning_Eta.png");

  GenMuon_VariableBins_Pt->Write();   GenMuon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/GenMuon_VariableBins_Pt.png");
  StandaloneMatchedME0Muon_VariableBins_Pt->Write();   StandaloneMatchedME0Muon_VariableBins_Pt->Draw();  c1->Print(histoFolder+"/StandaloneMatchedME0Muon_VariableBins_Pt.png");

  StandaloneMatchedME0Muon_Eta->Write();   StandaloneMatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/StandaloneMatchedME0Muon_Eta.png");
  StandaloneMatchedME0Muon_WideBinning_Eta->Write();   StandaloneMatchedME0Muon_WideBinning_Eta->Draw();  c1->Print(histoFolder+"/StandaloneMatchedME0Muon_WideBinning_Eta.png");
  StandaloneMatchedME0Muon_WidestBinning_Eta->Write();   StandaloneMatchedME0Muon_WidestBinning_Eta->Draw();  c1->Print(histoFolder+"/StandaloneMatchedME0Muon_WidestBinning_Eta.png");

 std::stringstream PtCutString;

  PtCutString<<"#splitline{Single #mu }{Reco Track p_{T} > "<<FakeRatePtCut<<" GeV}";
  const std::string& ptmp = PtCutString.str();
  const char* pcstr = ptmp.c_str();


  TLatex* txt =new TLatex;
  //txt->SetTextAlign(12);
  //txt->SetTextFont(42);
  txt->SetNDC();
  //txt->SetTextSize(0.05);
  txt->SetTextFont(132);
  txt->SetTextSize(0.05);


  // float t = c1->GetTopMargin();
  // float r = c1->GetRightMargin();

  // TLatex* latex1 = new TLatex;
  // latex1->SetNDC();
  // latex1->SetTextAngle(0);
  // latex1->SetTextColor(kBlack);    


  // latex1->SetTextFont(42);
  // latex1->SetTextAlign(31); 
  // //latex1->SetTextSize(lumiTextSize*t);    
  // latex1->SetTextSize(lumiTextSize);    
  // latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  TLatex* latex = new TLatex;
  latex->SetTextFont(cmsTextFont);
  latex->SetNDC();
  latex->SetTextSize(cmsTextSize);
  //latex->SetTextAlign(align_);

  // //End captions/labels
  

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
