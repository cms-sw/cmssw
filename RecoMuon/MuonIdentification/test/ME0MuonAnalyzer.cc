#include "FWCore/Framework/interface/Event.h"

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
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


//Associator for chi2: Including header files
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"

#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"

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
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>


#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TGraph.h"

#include <sstream>    

#include <iostream>
#include <fstream>

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
  virtual void endJob();
  //virtual void beginJob(const edm::EventSetup&);
  void beginJob();

  //For Track Association



  //protected:
  
  private:
  //Associator for chi2: objects
  //edm::InputTag associatormap;
  bool UseAssociators;
  bool RejectEndcapMuons;
  const TrackAssociatorByChi2* associatorByChi2;
  std::vector<std::string> associators;
  std::vector<const TrackAssociatorBase*> associator;
  std::vector<edm::InputTag> label;
  GenParticleCustomSelector gpSelector;	
  //std::string parametersDefiner;


  TString histoFolder;
  TFile* histoFile; 
  TH1F *Candidate_Eta;  TH1F *Mass_h; 
  TH1F *Segment_Eta;    TH1F *Segment_Phi;    TH1F *Segment_R;  TH2F *Segment_Pos;  
  TH1F *Rechit_Eta;    TH1F *Rechit_Phi;    TH1F *Rechit_R;  TH2F *Rechit_Pos;  
  TH1F *GenMuon_Phi;    TH1F *GenMuon_R;  TH2F *GenMuon_Pos;  
  TH1F *Track_Eta; TH1F *Track_Pt;  TH1F *ME0Muon_Eta; TH1F *ME0Muon_Pt;  TH1F *CheckME0Muon_Eta; 
  TH1F *ME0Muon_Cuts_Eta_5_10; TH1F *ME0Muon_Cuts_Eta_10_20; TH1F *ME0Muon_Cuts_Eta_20_40; TH1F *ME0Muon_Cuts_Eta_40; 
  TH1F *UnmatchedME0Muon_Eta; TH1F *UnmatchedME0Muon_Pt;    TH1F *Chi2UnmatchedME0Muon_Eta; 
  TH1F *UnmatchedME0Muon_Cuts_Eta_5_10;  TH1F *UnmatchedME0Muon_Cuts_Eta_10_20;  TH1F *UnmatchedME0Muon_Cuts_Eta_20_40;  TH1F *UnmatchedME0Muon_Cuts_Eta_40;
  TH1F *TracksPerSegment_h;  TH2F *TracksPerSegment_s;  TProfile *TracksPerSegment_p;
  TH2F *ClosestDelR_s; TProfile *ClosestDelR_p;
  TH2F *PtDiff_s; TProfile *PtDiff_p; TH1F *PtDiff_h; TH1F *QOverPtDiff_h; TH1F *PtDiff_rms; TH1F *PtDiff_gaus_narrow; TH1F *PtDiff_gaus_wide;
  TH1F *PtDiff_gaus_5_10;  TH1F *PtDiff_gaus_10_20;  TH1F *PtDiff_gaus_20_40; TH1F *PtDiff_gaus_40;
  TH1F *VertexDiff_h;
  TH2F *PDiff_s; TProfile *PDiff_p; TH1F *PDiff_h;
  TH2F *PtDiff_s_5_10;    TH2F *PtDiff_s_10_20;    TH2F *PtDiff_s_20_40;    TH2F *PtDiff_s_40;
  TH1F *FakeTracksPerSegment_h;  TH2F *FakeTracksPerSegment_s;  TProfile *FakeTracksPerSegment_p;
  TH1F *FakeTracksPerAssociatedSegment_h;  TH2F *FakeTracksPerAssociatedSegment_s;  TProfile *FakeTracksPerAssociatedSegment_p;
  TH1F *GenMuon_Eta; TH1F *GenMuon_Pt;   TH1F *MatchedME0Muon_Eta; TH1F *MatchedME0Muon_Pt; TH1F *Chi2MatchedME0Muon_Eta; TH1F *Chi2MatchedME0Muon_Pt; 
  TH1F *MatchedME0Muon_Eta_5_10;  TH1F *MatchedME0Muon_Eta_10_20;  TH1F *MatchedME0Muon_Eta_20_40;  TH1F *MatchedME0Muon_Eta_40;
  TH1F *GenMuon_Eta_5_10;  TH1F *GenMuon_Eta_10_20;  TH1F *GenMuon_Eta_20_40;  TH1F *GenMuon_Eta_40;
  TH1F *MuonRecoEff_Eta;  TH1F *MuonRecoEff_Pt;   TH1F *Chi2MuonRecoEff_Eta;  
  TH1F *MuonRecoEff_Eta_5_10;  TH1F *MuonRecoEff_Eta_10_20;  TH1F *MuonRecoEff_Eta_20_40;  TH1F *MuonRecoEff_Eta_40;
  TH1F *FakeRate_Eta;  TH1F *FakeRate_Pt;  TH1F *FakeRate_Eta_PerEvent;    TH1F *Chi2FakeRate_Eta;  
  TH1F *FakeRate_Eta_5_10;  TH1F *FakeRate_Eta_10_20;  TH1F *FakeRate_Eta_20_40;  TH1F *FakeRate_Eta_40;
  TH1F *MuonAllTracksEff_Eta;  TH1F *MuonAllTracksEff_Pt;
  TH1F *MuonUnmatchedTracksEff_Eta;  TH1F *MuonUnmatchedTracksEff_Pt; TH1F *FractionMatched_Eta;

  TH1F *UnmatchedME0Muon_Cuts_Eta;TH1F *ME0Muon_Cuts_Eta;

  TH1F *DelR_Segment_GenMuon;

  TH1F *SegPosDirPhiDiff_True_h;    TH1F *SegPosDirEtaDiff_True_h;     TH1F *SegPosDirPhiDiff_All_h;    TH1F *SegPosDirEtaDiff_All_h;   
  TH1F *SegTrackDirPhiDiff_True_h;    TH1F *SegTrackDirEtaDiff_True_h;     TH1F *SegTrackDirPhiDiff_All_h;    TH1F *SegTrackDirEtaDiff_All_h;   TH1F *SegTrackDirPhiPull_True_h;   TH1F *SegTrackDirPhiPull_All_h;   

  TH1F *SegGenDirPhiDiff_True_h;    TH1F *SegGenDirEtaDiff_True_h;     TH1F *SegGenDirPhiDiff_All_h;    TH1F *SegGenDirEtaDiff_All_h;   TH1F *SegGenDirPhiPull_True_h;   TH1F *SegGenDirPhiPull_All_h;   

  TH1F *XDiff_h;   TH1F *YDiff_h;   TH1F *XPull_h;   TH1F *YPull_h;



  TH1F *NormChi2_h;    TH1F *NormChi2Prob_h; TH2F *NormChi2VsHits_h;	TH2F *chi2_vs_eta_h;  TH1F *AssociatedChi2_h;  TH1F *AssociatedChi2_Prob_h;

  double  FakeRatePtCut, MatchingWindowDelR;

  double Nevents;


//Removing this
};

ME0MuonAnalyzer::ME0MuonAnalyzer(const edm::ParameterSet& iConfig) 
{
  histoFile = new TFile(iConfig.getParameter<std::string>("HistoFile").c_str(), "recreate");
  histoFolder = iConfig.getParameter<std::string>("HistoFolder").c_str();
  RejectEndcapMuons = iConfig.getParameter< bool >("RejectEndcapMuons");
  UseAssociators = iConfig.getParameter< bool >("UseAssociators");

  FakeRatePtCut   = iConfig.getParameter<double>("FakeRatePtCut");
  MatchingWindowDelR   = iConfig.getParameter<double>("MatchingWindowDelR");

  //Associator for chi2: getting parametters
  //associatormap = iConfig.getParameter< edm::InputTag >("associatormap");
  UseAssociators = iConfig.getParameter< bool >("UseAssociators");
  associators = iConfig.getParameter< std::vector<std::string> >("associators");

  label = iConfig.getParameter< std::vector<edm::InputTag> >("label"),

  gpSelector = GenParticleCustomSelector(iConfig.getParameter<double>("ptMinGP"),
					 iConfig.getParameter<double>("minRapidityGP"),
					 iConfig.getParameter<double>("maxRapidityGP"),
					 iConfig.getParameter<double>("tipGP"),
					 iConfig.getParameter<double>("lipGP"),
					 iConfig.getParameter<bool>("chargedOnlyGP"),
					 iConfig.getParameter<int>("statusGP"),
					 iConfig.getParameter<std::vector<int> >("pdgIdGP"));
  //parametersDefiner =iConfig.getParameter<std::string>("parametersDefiner");


}



//void ME0MuonAnalyzer::beginJob(const edm::EventSetup& iSetup)
void ME0MuonAnalyzer::beginJob()
{
  
  Candidate_Eta = new TH1F("Candidate_Eta"      , "Candidate #eta"   , 4, 2.0, 2.8 );

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

  //  GenMuon_Eta = new TH1F("GenMuon_Eta"      , "GenMuon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Phi = new TH1F("GenMuon_Phi"      , "GenMuon #phi"   , 60, -3, 3. );
  GenMuon_R = new TH1F("GenMuon_R"      , "GenMuon r"   , 30, 0, 150 );
  GenMuon_Pos = new TH2F("GenMuon_Pos"      , "GenMuon x,y"   ,100,-100.,100., 100,-100.,100. );

  ME0Muon_Eta = new TH1F("ME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_5_10 = new TH1F("ME0Muon_Cuts_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_10_20 = new TH1F("ME0Muon_Cuts_Eta_10_20"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_20_40 = new TH1F("ME0Muon_Cuts_Eta_20_40"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta_40 = new TH1F("ME0Muon_Cuts_Eta_40"      , "Muon #eta"   , 4, 2.0, 2.8 );

  CheckME0Muon_Eta = new TH1F("CheckME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Pt = new TH1F("ME0Muon_Pt"      , "Muon p_{T}"   , 120,0 , 120. );

  GenMuon_Eta = new TH1F("GenMuon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_5_10 = new TH1F("GenMuon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_10_20 = new TH1F("GenMuon_Eta_10_20"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_20_40 = new TH1F("GenMuon_Eta_20_40"      , "Muon #eta"   , 4, 2.0, 2.8 );
  GenMuon_Eta_40 = new TH1F("GenMuon_Eta_40"      , "Muon #eta"   , 4, 2.0, 2.8 );

  GenMuon_Pt = new TH1F("GenMuon_Pt"      , "Muon p_{T}"   , 120,0 , 120. );

  MatchedME0Muon_Eta = new TH1F("MatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_5_10 = new TH1F("MatchedME0Muon_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_10_20 = new TH1F("MatchedME0Muon_Eta_10_20"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_20_40 = new TH1F("MatchedME0Muon_Eta_20_40"      , "Muon #eta"   , 4, 2.0, 2.8 );
  MatchedME0Muon_Eta_40 = new TH1F("MatchedME0Muon_Eta_40"      , "Muon #eta"   , 4, 2.0, 2.8 );

  MatchedME0Muon_Pt = new TH1F("MatchedME0Muon_Pt"      , "Muon p_{T}"   , 40,0 , 20 );

  Chi2MatchedME0Muon_Eta = new TH1F("Chi2MatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  Chi2MatchedME0Muon_Pt = new TH1F("Chi2MatchedME0Muon_Pt"      , "Muon p_{T}"   , 40,0 , 20 );

  Chi2UnmatchedME0Muon_Eta = new TH1F("Chi2UnmatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );

  UnmatchedME0Muon_Eta = new TH1F("UnmatchedME0Muon_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_5_10 = new TH1F("UnmatchedME0Muon_Cuts_Eta_5_10"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_10_20 = new TH1F("UnmatchedME0Muon_Cuts_Eta_10_20"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_20_40 = new TH1F("UnmatchedME0Muon_Cuts_Eta_20_40"      , "Muon #eta"   , 4, 2.0, 2.8 );
  UnmatchedME0Muon_Cuts_Eta_40 = new TH1F("UnmatchedME0Muon_Cuts_Eta_40"      , "Muon #eta"   , 4, 2.0, 2.8 );

  UnmatchedME0Muon_Pt = new TH1F("UnmatchedME0Muon_Pt"      , "Muon p_{T}"   , 500,0 , 50 );

  UnmatchedME0Muon_Cuts_Eta = new TH1F("UnmatchedME0Muon_Cuts_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );
  ME0Muon_Cuts_Eta = new TH1F("ME0Muon_Cuts_Eta"      , "Muon #eta"   , 4, 2.0, 2.8 );

  Mass_h = new TH1F("Mass_h"      , "Mass"   , 100, 0., 200 );

  MuonRecoEff_Eta = new TH1F("MuonRecoEff_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );

  MuonRecoEff_Eta_5_10 = new TH1F("MuonRecoEff_Eta_5_10"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Eta_10_20 = new TH1F("MuonRecoEff_Eta_10_20"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Eta_20_40 = new TH1F("MuonRecoEff_Eta_20_40"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Eta_40 = new TH1F("MuonRecoEff_Eta_40"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  Chi2MuonRecoEff_Eta = new TH1F("Chi2MuonRecoEff_Eta"      , "Fraction of ME0Muons matched to gen muons"   ,4, 2.0, 2.8  );
  MuonRecoEff_Pt = new TH1F("MuonRecoEff_Pt"      , "Fraction of ME0Muons matched to gen muons"   ,8, 0,40  );

  FakeRate_Eta = new TH1F("FakeRate_Eta"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_5_10 = new TH1F("FakeRate_Eta_5_10"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_10_20 = new TH1F("FakeRate_Eta_10_20"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_20_40 = new TH1F("FakeRate_Eta_20_40"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );
  FakeRate_Eta_40 = new TH1F("FakeRate_Eta_40"      , "PU140, unmatched ME0Muons/all ME0Muons"   ,4, 2.0, 2.8  );

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
  
  DelR_Segment_GenMuon = new TH1F("DelR_Segment_GenMuon", "#Delta R between me0segment and gen muon",200,0,2);
  FractionMatched_Eta = new TH1F("FractionMatched_Eta"      , "Fraction of ME0Muons that end up successfully matched (matched/all)"   ,4, 2.0, 2.8  );

  PtDiff_s = new TH2F("PtDiff_s" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);

  PtDiff_s_5_10 = new TH2F("PtDiff_s_5_10" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);
  PtDiff_s_10_20 = new TH2F("PtDiff_s_10_20" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);
  PtDiff_s_20_40 = new TH2F("PtDiff_s_20_40" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);
  PtDiff_s_40 = new TH2F("PtDiff_s_40" , "Relative pt difference", 4, 2.0, 2.8, 200,-1,1.0);

  PtDiff_h = new TH1F("PtDiff_h" , "pt resolution", 100,-0.5,0.5);
  QOverPtDiff_h = new TH1F("QOverPtDiff_h" , "q/pt resolution", 100,-0.5,0.5);
  PtDiff_p = new TProfile("PtDiff_p" , "pt resolution vs. #eta", 4, 2.0, 2.8, -1.0,1.0,"s");

  PtDiff_rms    = new TH1F( "PtDiff_rms",    "RMS", 4, 2.0, 2.8 ); 
  PtDiff_gaus_wide    = new TH1F( "PtDiff_gaus_wide",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_narrow    = new TH1F( "PtDiff_gaus_narrow",    "GAUS_NARROW", 4, 2.0, 2.8 ); 

  PtDiff_gaus_5_10    = new TH1F( "PtDiff_gaus_5_10",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_10_20    = new TH1F( "PtDiff_gaus_10_20",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_20_40    = new TH1F( "PtDiff_gaus_20_40",    "GAUS_WIDE", 4, 2.0, 2.8 ); 
  PtDiff_gaus_40    = new TH1F( "PtDiff_gaus_40",    "GAUS_WIDE", 4, 2.0, 2.8 ); 

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



  XDiff_h = new TH1F("XDiff_h", "X Diff", 100, -10.0, 10.0 );
  YDiff_h = new TH1F("YDiff_h", "Y Diff", 100, -50.0, 50.0 ); 
  XPull_h = new TH1F("XPull_h", "X Pull", 100, -5.0, 5.0 );
  YPull_h = new TH1F("YPull_h", "Y Pull", 40, -50.0, 50.0 );

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

  //std::cout<<"ANALYZER"<<std::endl;
  
  using namespace edm;

  //run_ = (int)iEvent.id().run();
  //event_ = (int)iEvent.id().event();


    //David's functionality


  using namespace reco;

  // Handle <ME0MuonCollection > OurMuons;
  // iEvent.getByLabel <ME0MuonCollection> ("me0SegmentMatcher", OurMuons);

  Handle <std::vector<RecoChargedCandidate> > OurCandidates;
  iEvent.getByLabel <std::vector<RecoChargedCandidate> > ("me0MuonConverter", OurCandidates);

  //Handle<std::vector<EmulatedME0Segment> > OurSegments;
  //iEvent.getByLabel<std::vector<EmulatedME0Segment> >("me0SegmentProducer", OurSegments);

  Handle<GenParticleCollection> genParticles;

  iEvent.getByLabel<GenParticleCollection>("genParticles", genParticles);
  const GenParticleCollection genParticlesForChi2 = *(genParticles.product());

  unsigned int gensize=genParticles->size();

  if (RejectEndcapMuons){
    //Section to turn off signal muons in the endcaps, to approximate a nu gun
    for(unsigned int i=0; i<gensize; ++i) {
      const reco::GenParticle& CurrentParticle=(*genParticles)[i];
      if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
	if (fabs(CurrentParticle.eta()) > 1.9 ) {
	  std::cout<<"Found a signal muon outside the barrel, exiting the function"<<std::endl;
	  return;
	}
      }
    }      
  }



  Nevents++;


  Handle <TrackCollection > generalTracks;
  iEvent.getByLabel <TrackCollection> ("generalTracks", generalTracks);

  Handle <std::vector<ME0Muon> > OurMuons;
  iEvent.getByLabel <std::vector<ME0Muon> > ("me0SegmentMatcher", OurMuons);

  Handle<ME0SegmentCollection> OurSegments;
  iEvent.getByLabel("me0Segments","",OurSegments);


  edm::ESHandle<ME0Geometry> me0Geom;
  iSetup.get<MuonGeometryRecord>().get(me0Geom);

  ESHandle<MagneticField> bField;
  iSetup.get<IdealMagneticFieldRecord>().get(bField);
  ESHandle<Propagator> shProp;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAlong", shProp);
  


  //For Track Association:

  //std::cout<<"ON BEGIN JOB:"<<std::endl;
  if (UseAssociators) {
    //std::cout<<"Inside if:"<<std::endl;
    edm::ESHandle<TrackAssociatorBase> theAssociator;
    //std::cout<<"associators size = "<<associators.size()<<std::endl;
    for (unsigned int w=0;w<associators.size();w++) {
      //std::cout<<"On step "<<w<<std::endl;
      iSetup.get<TrackAssociatorRecord>().get(associators[w],theAssociator);
      //std::cout<<"On step "<<w<<std::endl;
      associator.push_back( theAssociator.product() );
      //std::cout<<"On step "<<w<<std::endl;
    }
    //std::cout<<"Got this many associators: "<<associator.size()<<std::endl;
  }

  //For Track Association:
  //edm::ESHandle<ParametersDefinerForTP> parametersDefinerTP; 
  //iSetup.get<TrackAssociatorRecord>().get(parametersDefiner,parametersDefinerTP);


  //=====Finding ME0Muons that match gen muons, plotting the closest of those
  //    -----First, make a vector of bools for each ME0Muon

  std::vector<bool> IsMatched;
  std::vector<int> SegIdForMatch;
  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon){
    IsMatched.push_back(false);
    SegIdForMatch.push_back(-1);
  }
  //std::cout<<IsMatched.size()<<" total me0muons"<<std::endl;
  //   -----Now, loop over each gen muon to compare it to each ME0Muon
  //   -----For each gen muon, take the closest ME0Muon that is a match within delR 0.15
  //   -----Each time a match on an ME0Muon is made, change the IsMatched bool corresponding to it to true
  //   -----Also, each time a match on an ME0Muon is made, we plot the pt and eta of the gen muon it was matched to

  // unsigned int gensize=genParticles->size();
  // for(unsigned int i=0; i<gensize; ++i) {
  //   const reco::GenParticle& CurrentParticle=(*genParticles)[i];
  //   if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
  //     for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
  // 	   thisTrack != generalTracks->end();++thisTrack){
  // 	VertexDiff_h->Fill(fabs(thisTrack->vz()-CurrentParticle.vz()));
  // 	PtDiff_h->Fill(fabs(thisTrack->pt() - CurrentParticle.pt())/CurrentParticle.pt());
  // 	PtDiff_s->Fill(CurrentParticle.eta(),fabs(thisTrack->pt() - CurrentParticle.pt())/CurrentParticle.pt());
  // 	PtDiff_p->Fill(CurrentParticle.eta(),fabs(thisTrack->pt() - CurrentParticle.pt())/CurrentParticle.pt());
  // 	PDiff_h->Fill(fabs(thisTrack->p() - CurrentParticle.p())/CurrentParticle.p());
  // 	PDiff_s->Fill(CurrentParticle.eta(),fabs(thisTrack->p() - CurrentParticle.p())/CurrentParticle.p());
  // 	PDiff_p->Fill(CurrentParticle.eta(),fabs(thisTrack->p() - CurrentParticle.p())/CurrentParticle.p());
  //     }
  //   }
  // }
  //--------------------------------------------------------------


  //============================Debugging
  //  bool FoundRightMuon = false;

  // for(unsigned int i=0; i<gensize; ++i) {
  //    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
  //    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
  //      if ((fabs(CurrentParticle.eta()) < 2.0 ) ||(fabs(CurrentParticle.eta()) > 2.8 )) continue;
  //      std::cout<<"Current Particle = "<<CurrentParticle.eta()<<", "<<CurrentParticle.phi()<<std::endl;
  //      for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
  // 	   thisMuon != OurMuons->end(); ++thisMuon){
  // 	TrackRef tkRef = thisMuon->innerTrack();
  // 	double thisDelR = reco::deltaR(CurrentParticle,*tkRef);

  // 	ME0Segment Seg = thisMuon->me0segment();
  // 	ME0DetId id =Seg.me0DetId();
  // 	auto roll = me0Geom->etaPartition(id); 
  // 	auto GlobVect(roll->toGlobal(Seg.localPosition()));

  // 	if (thisDelR < 0.15){

  // 	  std::cout<<"ME0Muon = "<<tkRef->pt()<<", "<<tkRef->eta()<<", "<<tkRef->phi()<<std::endl;
  // 	  std::cout<<"Segment = "<<GlobVect.eta()<<", "<<GlobVect.phi()<<std::endl;

	  
  // 	}
  // 	else if (tkRef->pt() > 8.0){
  // 	  std::cout<<"Out of range = "<<tkRef->pt()<<", "<<tkRef->eta()<<", "<<tkRef->phi()<<std::endl;
  // 	  std::cout<<"Segment = "<<GlobVect.eta()<<", "<<GlobVect.phi()<<std::endl;
  // 	}
  // 	if (fabs(tkRef->pt()-10.0) < 1.0) FoundRightMuon=true;
	
  //      }
  //    }
  // }


      
    
  // if (!FoundRightMuon){
  // std::cout<<"Current Particle = "<<CurrentParticle.eta()<<", "<<CurrentParticle.phi()<<std::endl;
  // std::cout<<"Pos:"<<std::endl;
  // std::cout<<r3Final.eta()<<", "<<r3Final.phi()<<std::endl;
  // 	}
  // for (auto thisSegment = OurSegments->begin(); thisSegment != OurSegments->end(); 
  // 	   ++thisSegment){
  // 	ME0DetId id = thisSegment->me0DetId();
  // 	//std::cout<<"ME0DetId =  "<<id<<std::endl;
  // 	auto roll = me0Geom->etaPartition(id); 
  // 	auto GlobVect(roll->toGlobal(thisSegment->localPosition()));

  // 	double thisDelR = reco::deltaR(r3Final,GlobVect);
  // 	if (!FoundRightMuon){
  // 	if (thisDelR < 0.3){
  // 	  std::cout<<"Segment = "<<GlobVect.eta()<<", "<<GlobVect.phi()<<std::endl;
  // 	  std::cout<<"local errors = "<<thisSegment->localPositionError().xx()<<", "<<thisSegment->localPositionError().yy()<<std::endl;
  // 	}
  // 	}


  // for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
  // 	   thisTrack != generalTracks->end();++thisTrack){

  // //Remove later
  // if (fabs(thisTrack->eta()) < 1.8) continue;
  // //if (fabs(thisTrack->pt()) < 0.6) continue;

  // //std::cout<<"Track pt = "<<thisTrack->pt();
  // float zSign  = thisTrack->pz()/fabs(thisTrack->pz());

  // //float zValue = 560. * zSign;
  // float zValue = 526.75 * zSign;
  // Plane *plane = new Plane(Surface::PositionType(0,0,zValue),Surface::RotationType());
  // //Getting the initial variables for propagation
  // int chargeReco = thisTrack->charge(); 
  // GlobalVector p3reco, r3reco;

  // p3reco = GlobalVector(thisTrack->outerPx(), thisTrack->outerPy(), thisTrack->outerPz());
  // r3reco = GlobalVector(thisTrack->outerX(), thisTrack->outerY(), thisTrack->outerZ());

  // AlgebraicSymMatrix66 covReco;
  // //This is to fill the cov matrix correctly
  // AlgebraicSymMatrix55 covReco_curv;
  // covReco_curv = thisTrack->outerStateCovariance();
  // FreeTrajectoryState initrecostate = getFTS(p3reco, r3reco, chargeReco, covReco_curv, &*bField);
  // getFromFTS(initrecostate, p3reco, r3reco, chargeReco, covReco);

  // //Now we propagate and get the propagated variables from the propagated state
  // SteppingHelixStateInfo startrecostate(initrecostate);
  // SteppingHelixStateInfo lastrecostate;

  // const SteppingHelixPropagator* ThisshProp = 
  // 	dynamic_cast<const SteppingHelixPropagator*>(&*shProp);
	
  // lastrecostate = ThisshProp->propagate(startrecostate, *plane);
	
  // FreeTrajectoryState finalrecostate;
  // lastrecostate.getFreeState(finalrecostate);
      
  // AlgebraicSymMatrix66 covFinalReco;
  // GlobalVector p3FinalReco_glob, r3FinalReco_globv;
  // getFromFTS(finalrecostate, p3FinalReco_glob, r3FinalReco_globv, chargeReco, covFinalReco);
      
  // GlobalPoint r3FinalReco_glob(r3FinalReco_globv.x(),r3FinalReco_globv.y(),r3FinalReco_globv.z());

  // for (auto thisSegment = OurSegments->begin(); thisSegment != OurSegments->end(); 
  // 	   ++thisSegment){
  // 	ME0DetId id = thisSegment->me0DetId();
  // 	auto roll = me0Geom->etaPartition(id); 
  // 	//auto GlobVect(roll->toGlobal(thisSegment->localPosition()));
	
  // 	LocalPoint r3FinalReco = roll->toLocal(r3FinalReco_glob);
  // 	LocalVector p3FinalReco=roll->toLocal(p3FinalReco_glob);
  // 	LocalTrajectoryParameters ltp(r3FinalReco,p3FinalReco,chargeReco);
  // 	JacobianCartesianToLocal jctl(roll->surface(),ltp);
  // 	AlgebraicMatrix56 jacobGlbToLoc = jctl.jacobian(); 

  // 	AlgebraicMatrix55 Ctmp =  (jacobGlbToLoc * covFinalReco) * ROOT::Math::Transpose(jacobGlbToLoc); 
  // 	AlgebraicSymMatrix55 C;  // I couldn't find any other way, so I resort to the brute force
  // 	for(int i=0; i<5; ++i) {
  // 	  for(int j=0; j<5; ++j) {
  // 	    C[i][j] = Ctmp[i][j]; 

  // 	  }
  // 	}  

  // 	Double_t sigmax = sqrt(C[3][3]+thisSegment->localPositionError().xx() );      
  // 	Double_t sigmay = sqrt(C[4][4]+thisSegment->localPositionError().yy() );



  // 	XPull_h->Fill((Seg->localPosition().x()-r3FinalReco.x())/sigmax);
  // 	YPull_h->Fill((Seg->localPosition().y()-r3FinalReco.y())/sigmay);

  // 	XDiff_h->Fill((Seg->localPosition().x()-r3FinalReco.x()));
  // 	YDiff_h->Fill((Seg->localPosition().y()-r3FinalReco.y()));
	

  // }
  // // double thisDelR = reco::deltaR(CurrentParticle,*thisTrack);
  // // if (false){

	
  // // }
  // //       if (!FoundRightMuon){
  // //  	if (thisDelR < 0.15){

  // //  	  std::cout<<"Track = "<<thisTrack->pt()<<", "<<thisTrack->eta()<<", "<<thisTrack->phi()<<std::endl;
  // //  	  std::cout<<"Pos:"<<std::endl;
  // //  	  std::cout<<r3FinalReco_globv.eta()<<", "<<r3FinalReco_globv.phi()<<std::endl;

  // //  	  std::cout<<"Err:"<<std::endl;
  // //  	  std::cout<<covFinalReco(0,0)<<", "<<covFinalReco(1,1)<<std::endl;
  // //  	}
  // //       }
  // //       }
  // // }
  // }

  //Track Association by Chi2:

  //int w=0;
  //std::cout<<"associators size = "<<associators.size()<<std::endl;
  for (unsigned int ww=0;ww<associators.size();ww++){

    //std::cout<<associator[ww]<<std::endl;
    //std::cout<<"Starting loop over associators (only 1 I think)"<<std::endl;

    associatorByChi2 = dynamic_cast<const TrackAssociatorByChi2*>(associator[ww]);

    //associatorByChi2 = associator[ww];
    //std::cout<<"here now"<<std::endl;
    //std::cout<<"associatorByChi2 = "<<associatorByChi2<<std::endl;

    if (associatorByChi2==0) continue;
    //if (associator[ww]==0) continue;
    //std::cout<<"here now"<<std::endl;

    //std::cout<<"label size = "<<label.size()<<std::endl;
    for (unsigned int www=0;www<label.size();www++){
      //
      reco::RecoToGenCollection recSimColl;
      reco::GenToRecoCollection simRecColl;
      edm::Handle<View<Track> >  trackCollection;



      unsigned int trackCollectionSize = 0;

      //      if(!event.getByLabel(label[www], trackCollection)&&ignoremissingtkcollection_) continue;
      //std::cout<<label[www].process()<<", "<<label[www].label()<<", "<<label[www].instance()<<std::endl;
      if(!iEvent.getByLabel(label[www], trackCollection)) {
	//std::cout<<"Inserting, since no label"<<std::endl;
	recSimColl.post_insert();
	simRecColl.post_insert();

      }

      else {
	trackCollectionSize = trackCollection->size();
	recSimColl=associatorByChi2->associateRecoToGen(trackCollection,
							genParticles,
							&iEvent,
							&iSetup);
	//std::cout<<"here now"<<std::endl;
	simRecColl=associatorByChi2->associateGenToReco(trackCollection,
							genParticles,
							&iEvent,
							&iSetup);
      }
      //std::cout<<"here now"<<std::endl;
      //int ats = 0;
      //std::cout<<"genParticlesForChi2.size() = "<<genParticlesForChi2.size()<<std::endl;
      for (GenParticleCollection::size_type i=0; i<genParticlesForChi2.size(); i++){
	//bool TP_is_matched = false;
	double quality = 0.;
	//bool Quality05  = false;
	//bool Quality075 = false;

	GenParticleRef tpr(genParticles, i);
	GenParticle* tp=const_cast<GenParticle*>(tpr.get());
	TrackingParticle::Vector momentumTP; 
	TrackingParticle::Point vertexTP;
	//double dxySim = 0;
	//double dzSim = 0;

	//Collision like particle
	if(! gpSelector(*tp)) continue;
	momentumTP = tp->momentum();
	vertexTP = tp->vertex();
	
	std::vector<std::pair<RefToBase<Track>, double> > rt;

	//Check if the gen particle has been associated to any reco track
	if(simRecColl.find(tpr) != simRecColl.end()){
	  rt = (std::vector<std::pair<RefToBase<Track>, double> >) simRecColl[tpr];
	  //It has, so we check that the pair TrackRef/double pair collection (vector of pairs) is not empty
	  if (rt.size()!=0) {
	    //It is not empty, so there is at least one real track that the gen particle is matched to
	    
	    //We take the first element of the vector, .begin(), and the trackRef from it, ->first, this is our best possible track match
	    RefToBase<Track> assoc_recoTrack = rt.begin()->first;
	    std::cout<<"-----------------------------associated Track #"<<assoc_recoTrack.key()<<std::endl;

	    quality = rt.begin()->second;
	    std::cout << "TrackingParticle #" <<tpr.key()  
		      << " with pt=" << sqrt(momentumTP.perp2()) 
		      << " associated with quality:" << quality <<std::endl;

	    //Also, seeing as we have found a gen particle that is matched to a track, it is efficient, so we put it in the numerator of the efficiency plot
	    //if (( sqrt(momentumTP.perp2()) > FakeRatePtCut) && (TMath::Abs(tp->eta()) < 2.8) )Chi2MatchedME0Muon_Eta->Fill(tp->eta());
	    //if ( ( assoc_recoTrack->pt() > FakeRatePtCut) && (TMath::Abs(tp->eta()) < 2.8) )Chi2MatchedME0Muon_Eta->Fill(tp->eta());

	  }

	}//END if(simRecColl.find(tpr) != simRecColl.end())
      }//END for (GenParticleCollection::size_type i=0; i<genParticlesForChi2.size(); i++)

      
      for(View<Track>::size_type i=0; i<trackCollectionSize; ++i){
	//bool Track_is_matched = false; 
	RefToBase<Track> track(trackCollection, i);

	//std::vector<std::pair<TrackingParticleRef, double> > tp;
	std::vector<std::pair<GenParticleRef, double> > tp;
	std::vector<std::pair<GenParticleRef, double> > tpforfake;
	//TrackingParticleRef tpr;
	GenParticleRef tpr;
	GenParticleRef tprforfake;

	//Check if the track is associated to any gen particle
	if(recSimColl.find(track) != recSimColl.end()){
	  
	  tp = recSimColl[track];
	  if (tp.size()!=0) {
	    //Track_is_matched = true;
	    tpr = tp.begin()->first;

	    double assocChi2 = -(tp.begin()->second);
	   
	    //So this track is matched to a gen particle, lets get that gen particle now
	    if (  (simRecColl.find(tpr) != simRecColl.end())    ){
	      std::vector<std::pair<RefToBase<Track>, double> > rt;
	      std::cout<<"Comparing gen and reco tracks"<<std::endl;
	      if  (simRecColl[tpr].size() > 0){
		rt=simRecColl[tpr];
		RefToBase<Track> bestrecotrackforeff = rt.begin()->first;
		//Only fill the efficiency histo if the track found matches up to a gen particle's best choice
		if (bestrecotrackforeff == track) {
		  if ( (track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) )Chi2MatchedME0Muon_Eta->Fill(fabs(tpr->eta()));
		  if ( (track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) )AssociatedChi2_h->Fill(assocChi2);
		  if ( (track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) )AssociatedChi2_Prob_h->Fill(TMath::Prob((assocChi2)*5,5));
		  std::cout<<"assocChi2 = "<<assocChi2<<std::endl;
		}
	      }
	    }
	  }
	}
	//End checking of Efficient muons

	//For Fakes --------------------------------------------

	
	//Check if finding a track associated to a gen particle fails, or if there is no track in the collection at all

	if( (recSimColl.find(track) == recSimColl.end() ) || ( recSimColl[track].size() == 0  ) ){
	  //So we've now determined that this track is not associated to any gen, and fill our histo of fakes:
	  if ((track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) ) Chi2UnmatchedME0Muon_Eta->Fill(fabs(track->eta()));
	}
	
	//Its possible that the track is associated to a gen particle, but isn't the best match and would still fail
	//In that case, we go to the gen particle...
	else if (recSimColl[track].size() > 0){
	  tpforfake = recSimColl[track];
	  tprforfake=tpforfake.begin()->first;
	  //We now have the gen particle, to check

	  //If for some crazy reason we can't find the gen particle, that means its a fake
	  if (  (simRecColl.find(tprforfake) == simRecColl.end())  ||  (simRecColl[tprforfake].size() == 0)  ) {
	    if ((track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8))  Chi2UnmatchedME0Muon_Eta->Fill(fabs(track->eta()));
	  }
	  //We can probably find the gen particle
	  else if(simRecColl[tprforfake].size() > 0)  {
	    //We can now access the best possible track for the gen particle that this track was matched to
	    std::vector<std::pair<RefToBase<Track>, double> > rtforfake;
	    rtforfake=simRecColl[tprforfake];
	   
	    RefToBase<Track> bestrecotrack = rtforfake.begin()->first;
	    //if the best reco track is NOT the track that we're looking at, we know we have a fake, that was within the cut, but not the closest
	    if (bestrecotrack != track) {
	      if ( (track->pt() > FakeRatePtCut) && (TMath::Abs(track->eta()) < 2.8) ) Chi2UnmatchedME0Muon_Eta->Fill(fabs(track->eta()));
	    }

	  }
	}

	//End For Fakes --------------------------------------------

	
	if (TMath::Abs(track->eta()) < 2.8) CheckME0Muon_Eta->Fill(fabs(track->eta()));	

	NormChi2_h->Fill(track->normalizedChi2());
	NormChi2Prob_h->Fill(TMath::Prob(track->chi2(),(int)track->ndof()));
	NormChi2VsHits_h->Fill(track->numberOfValidHits(),track->normalizedChi2());


	chi2_vs_eta_h->Fill((track->eta()),track->normalizedChi2());

	//nhits_vs_eta_h->Fill((track->eta()),track->numberOfValidHits());


      }//END for(View<Track>::size_type i=0; i<trackCollectionSize; ++i){
    }//END for (unsigned int www=0;www<label.size();www++)
  }// END for (unsigned int www=0;www<label.size();www++)
  std::cout<<"Finished chi2 stuff"<<std::endl;
  std::vector<int> MatchedSegIds;

  for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  

      //if ( (fabs(CurrentParticle.eta()) > 2.0) && (fabs(CurrentParticle.eta()) < 2.8) )  GenMuon_Eta->Fill(CurrentParticle.eta());
      //std::cout<<"Mother's ID is: "<<CurrentParticle.motherId()<<std::endl;
     
      double LowestDelR = 9999;
      double thisDelR = 9999;
      int MatchedID = -1;
      int ME0MuonID = 0;

      std::vector<double> ReferenceTrackPt;

      double VertexDiff=-1,PtDiff=-1,QOverPtDiff=-1,PDiff=-1;

      std::cout<<"Size = "<<OurMuons->size()<<std::endl;
      for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
	   thisMuon != OurMuons->end(); ++thisMuon){
	TrackRef tkRef = thisMuon->innerTrack();
	SegIdForMatch.push_back(thisMuon->me0segid());
	thisDelR = reco::deltaR(CurrentParticle,*tkRef);
	ReferenceTrackPt.push_back(tkRef->pt());
	if (tkRef->pt() > FakeRatePtCut ) {
	  if (thisDelR < MatchingWindowDelR ){
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

      if (MatchedID != -1){
	IsMatched[MatchedID] = true;
	if (CurrentParticle.pt() >FakeRatePtCut) {
	  MatchedME0Muon_Eta->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 5.0) && (CurrentParticle.pt() <= 10.0) )  	MatchedME0Muon_Eta_5_10->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 10.0) && (CurrentParticle.pt() <= 20.0) )	MatchedME0Muon_Eta_10_20->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 20.0) && (CurrentParticle.pt() <= 40.0) )	MatchedME0Muon_Eta_20_40->Fill(fabs(CurrentParticle.eta()));
	  if ( CurrentParticle.pt() > 40.0) 		MatchedME0Muon_Eta_40->Fill(fabs(CurrentParticle.eta()));
	


	  VertexDiff_h->Fill(VertexDiff);
	  PtDiff_h->Fill(PtDiff);	
	  QOverPtDiff_h->Fill(QOverPtDiff);
	  PtDiff_s->Fill(CurrentParticle.eta(),PtDiff);
	  if ( (CurrentParticle.pt() > 5.0) && (CurrentParticle.pt() <= 10.0) ) 	PtDiff_s_5_10->Fill(CurrentParticle.eta(),PtDiff);
	  if ( (CurrentParticle.pt() > 10.0) && (CurrentParticle.pt() <= 20.0) )	PtDiff_s_10_20->Fill(CurrentParticle.eta(),PtDiff);
	  if ( (CurrentParticle.pt() > 20.0) && (CurrentParticle.pt() <= 40.0) )	PtDiff_s_20_40->Fill(CurrentParticle.eta(),PtDiff);
	  if ( CurrentParticle.pt() > 40.0) 	PtDiff_s_40->Fill(CurrentParticle.eta(),PtDiff);
	  PtDiff_p->Fill(CurrentParticle.eta(),PtDiff);
	
	  PDiff_h->Fill(PDiff);
	  PDiff_s->Fill(CurrentParticle.eta(),PDiff);
	  PDiff_p->Fill(CurrentParticle.eta(),PDiff);
	}
	MatchedSegIds.push_back(SegIdForMatch[MatchedID]);

	// if ( ((CurrentParticle.eta()) > 2.4) && ((CurrentParticle.eta()) < 3.8) ) {
	//   //MatchedME0Muon_Pt->Fill(CurrentParticle.pt());
	//   //MatchedME0Muon_Pt->Fill(.pt());
	  
	// }
      }
	if (CurrentParticle.pt() >FakeRatePtCut) {
	  GenMuon_Eta->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 5.0) && (CurrentParticle.pt() <= 10.0) )  	GenMuon_Eta_5_10->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 10.0) && (CurrentParticle.pt() <= 20.0) )	GenMuon_Eta_10_20->Fill(fabs(CurrentParticle.eta()));
	  if ( (CurrentParticle.pt() > 20.0) && (CurrentParticle.pt() <= 40.0) )	GenMuon_Eta_20_40->Fill(fabs(CurrentParticle.eta()));
	  if ( CurrentParticle.pt() > 40.0) 		GenMuon_Eta_40->Fill(fabs(CurrentParticle.eta()));
	  GenMuon_Phi->Fill(CurrentParticle.phi());
	  if ( ((CurrentParticle.eta()) > 2.0) && ((CurrentParticle.eta()) < 2.8) ) GenMuon_Pt->Fill(CurrentParticle.pt());
	}

    }
  }


//Diff studies for gen matching

  std::cout<<"Doing first propagation"<<std::endl;
 for(unsigned int i=0; i<gensize; ++i) {
    const reco::GenParticle& CurrentParticle=(*genParticles)[i];
    if ( (CurrentParticle.status()==1) && ( (CurrentParticle.pdgId()==13)  || (CurrentParticle.pdgId()==-13) ) ){  
      if ((fabs(CurrentParticle.eta()) < 2.0 ) ||(fabs(CurrentParticle.eta()) > 2.8 )) continue;

      float zSign  = CurrentParticle.pz()/fabs(CurrentParticle.pz());

    	float zValue = 526.75 * zSign;
    	Plane *plane = new Plane(Surface::PositionType(0,0,zValue),Surface::RotationType());
    	TLorentzVector Momentum;
    	Momentum.SetPtEtaPhiM(CurrentParticle.pt()
    			      ,CurrentParticle.eta()
    			      ,CurrentParticle.phi()
    			      ,CurrentParticle.mass());
    	GlobalVector p3gen(Momentum.Px(), Momentum.Py(), Momentum.Pz());
    	GlobalVector r3gen = GlobalVector(CurrentParticle.vertex().x()
    					  ,CurrentParticle.vertex().y()
    					  ,CurrentParticle.vertex().z());

    	AlgebraicSymMatrix66 covGen = AlgebraicMatrixID(); 
    	covGen *= 1e-20; // initialize to sigma=1e-10 .. should get overwhelmed by MULS
    	AlgebraicSymMatrix66 covFinal;
    	int chargeGen =  CurrentParticle.charge(); 

    	//Propagation
    	FreeTrajectoryState initstate = getFTS(p3gen, r3gen, chargeGen, covGen, &*bField);
	
    	SteppingHelixStateInfo startstate(initstate);
    	SteppingHelixStateInfo laststate;

    	const SteppingHelixPropagator* ThisshProp = 
    	  dynamic_cast<const SteppingHelixPropagator*>(&*shProp);

    	laststate = ThisshProp->propagate(startstate, *plane);

    	FreeTrajectoryState finalstate;
    	laststate.getFreeState(finalstate);
	
    	GlobalVector p3Final, r3Final;
    	getFromFTS(finalstate, p3Final, r3Final, chargeGen, covFinal);


	int ME0MuonID = 0;

	for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
	   thisMuon != OurMuons->end(); ++thisMuon){

	TrackRef tkRef = thisMuon->innerTrack();
	SegIdForMatch.push_back(thisMuon->me0segid());

	ME0Segment Seg = thisMuon->me0segment();
	ME0DetId id =Seg.me0DetId();
	auto roll = me0Geom->etaPartition(id); 

	double DirectionPull, DirectionPullNum, DirectionPullDenom;

	//Computing the sigma for the track direction
	Double_t mag_track = p3Final.perp();
	//Double_t phi_track = p3Final.phi();

	//Double_t dmagdx_track = p3Final.x()/mag_track;
	//Double_t dmagdy_track = p3Final.y()/mag_track;
	Double_t dphidx_track = -p3Final.y()/(mag_track*mag_track);
	Double_t dphidy_track = p3Final.x()/(mag_track*mag_track);
	Double_t sigmaphi_track = sqrt( dphidx_track*dphidx_track*covFinal(3,3)+
					dphidy_track*dphidy_track*covFinal(4,4)+
					dphidx_track*dphidy_track*2*covFinal(3,4) );

	DirectionPullNum = p3Final.phi()-roll->toGlobal(Seg.localDirection()).phi();
	DirectionPullDenom = sqrt( pow(roll->toGlobal(Seg.localPosition()).phi(),2) + pow(sigmaphi_track,2) );
	DirectionPull = DirectionPullNum / DirectionPullDenom;
	

	if (IsMatched[ME0MuonID]){
	  SegGenDirPhiDiff_True_h->Fill(p3Final.phi()-roll->toGlobal(Seg.localDirection()).phi() );
	  SegGenDirEtaDiff_True_h->Fill(p3Final.eta()-roll->toGlobal(Seg.localDirection()).eta() );
	  SegGenDirPhiPull_True_h->Fill(DirectionPull);
	}

	if ((zSign * roll->toGlobal(Seg.localDirection()).z()) > 0 ){
	  SegGenDirPhiDiff_All_h->Fill(p3Final.phi()-roll->toGlobal(Seg.localDirection()).phi() );
	  SegGenDirPhiPull_All_h->Fill(DirectionPull);
    
	  SegGenDirEtaDiff_All_h->Fill(p3Final.eta()-roll->toGlobal(Seg.localDirection()).eta() );
	}
	ME0MuonID++;
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
    TrackRef tkRef = thisMuon->innerTrack();


    if (IsMatched[ME0MuonID]) {
      if ( (TMath::Abs(tkRef->eta()) > 2.0) && (TMath::Abs(tkRef->eta()) < 2.8) )  MatchedME0Muon_Pt->Fill(tkRef->pt());

      //Moved resolution stuff here, only calculate resolutions for matched muons!


    }

    if (!IsMatched[ME0MuonID]){

      UnmatchedME0Muon_Eta->Fill(fabs(tkRef->eta()));
      if ((tkRef->pt() > FakeRatePtCut) && (TMath::Abs(tkRef->eta()) < 2.8) )  {
	if ( (tkRef->pt() > 5.0) && (tkRef->pt() <= 10.0) )  	UnmatchedME0Muon_Cuts_Eta_5_10->Fill(fabs(tkRef->eta()));
	if ( (tkRef->pt() > 10.0) && (tkRef->pt() <= 20.0) )	UnmatchedME0Muon_Cuts_Eta_10_20->Fill(fabs(tkRef->eta()));
	if ( (tkRef->pt() > 20.0) && (tkRef->pt() <= 40.0) )	UnmatchedME0Muon_Cuts_Eta_20_40->Fill(fabs(tkRef->eta()));
	if ( tkRef->pt() > 40.0) 		UnmatchedME0Muon_Cuts_Eta_40->Fill(fabs(tkRef->eta()));

	UnmatchedME0Muon_Cuts_Eta->Fill(fabs(tkRef->eta()));
      }
      //if ( (TMath::Abs(tkRef->eta()) > 2.0) && (TMath::Abs(tkRef->eta()) < 3.4) ) UnmatchedME0Muon_Pt->Fill(tkRef->pt());
      if ( (TMath::Abs(tkRef->eta()) < 2.8) ) UnmatchedME0Muon_Pt->Fill(tkRef->pt());
    }
    ME0MuonID++;
  }
  


  // for (std::vector<ME0Segment>::const_iterator thisSegment = OurSegments->begin();
  //      thisSegment != OurSegments->end();++thisSegment){
  //   LocalVector TempVect(thisSegment->localDirection().x(),thisSegment->localDirection().y(),thisSegment->localDirection().z());
  //   Segment_Eta->Fill(TempVect.eta());
  // }

  
  for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
       thisTrack != generalTracks->end();++thisTrack){
    Track_Eta->Fill(fabs(thisTrack->eta()));
    if ( (TMath::Abs(thisTrack->eta()) > 2.0) && (TMath::Abs(thisTrack->eta()) < 2.8) ) Track_Pt->Fill(thisTrack->pt());

    

    
  }

  // for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
  //      thisMuon != OurMuons->end(); ++thisMuon){
  //   TrackRef tkRef = thisMuon->innerTrack();
  //   ME0Muon_Eta->Fill(tkRef->eta());
  //   if ( (TMath::Abs(tkRef->eta()) > 2.4) && (TMath::Abs(tkRef->eta()) < 4.0) ) ME0Muon_Pt->Fill(tkRef->pt());
  // }
  
  std::vector<double> SegmentEta, SegmentPhi, SegmentR, SegmentX, SegmentY;
  // std::vector<const ME0Segment*> Ids;
  // std::vector<const ME0Segment*> Ids_NonGenMuons;
  // std::vector<const ME0Segment*> UniqueIdList;
   std::vector<int> Ids;
   std::vector<int> Ids_NonGenMuons;
   std::vector<int> UniqueIdList;
   int TrackID=0;

   std::cout<<"Doing some propagation"<<std::endl;
   int MuID = 0;
   for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
	thisMuon != OurMuons->end(); ++thisMuon){
    TrackRef tkRef = thisMuon->innerTrack();
    //ME0Segment segRef = thisMuon->me0segment();
    //const ME0Segment* SegId = segRef->get();

    ME0Segment Seg = thisMuon->me0segment();
    ME0DetId id =Seg.me0DetId();
    auto roll = me0Geom->etaPartition(id); 
    auto GlobVect(roll->toGlobal(Seg.localPosition()));

    int SegId=thisMuon->me0segid();

    //std::cout<<SegId<<std::endl;


    //For a direction study...
    if ( (tkRef->pt() > 5.0) && (IsMatched[MuID]) ){
      SegPosDirPhiDiff_True_h->Fill(roll->toGlobal(Seg.localPosition()).phi()-roll->toGlobal(Seg.localDirection()).phi() );
      SegPosDirEtaDiff_True_h->Fill(roll->toGlobal(Seg.localPosition()).eta()-roll->toGlobal(Seg.localDirection()).eta() );
    }

    SegPosDirPhiDiff_All_h->Fill(roll->toGlobal(Seg.localPosition()).phi()-roll->toGlobal(Seg.localDirection()).phi() );
    SegPosDirEtaDiff_All_h->Fill(roll->toGlobal(Seg.localPosition()).eta()-roll->toGlobal(Seg.localDirection()).eta() );

    // For another direction study...
    float zSign  = tkRef->pz()/fabs(tkRef->pz());

    //float zValue = 560. * zSign;
    float zValue = 526.75 * zSign;
    Plane *plane = new Plane(Surface::PositionType(0,0,zValue),Surface::RotationType());
    //Getting the initial variables for propagation
    int chargeReco = tkRef->charge(); 
    GlobalVector p3reco, r3reco;

    p3reco = GlobalVector(tkRef->outerPx(), tkRef->outerPy(), tkRef->outerPz());
    r3reco = GlobalVector(tkRef->outerX(), tkRef->outerY(), tkRef->outerZ());

    AlgebraicSymMatrix66 covReco;
    //This is to fill the cov matrix correctly
    AlgebraicSymMatrix55 covReco_curv;
    covReco_curv = tkRef->outerStateCovariance();
    FreeTrajectoryState initrecostate = getFTS(p3reco, r3reco, chargeReco, covReco_curv, &*bField);
    getFromFTS(initrecostate, p3reco, r3reco, chargeReco, covReco);

    //Now we propagate and get the propagated variables from the propagated state
    SteppingHelixStateInfo startrecostate(initrecostate);
    SteppingHelixStateInfo lastrecostate;

    const SteppingHelixPropagator* ThisshProp = 
      dynamic_cast<const SteppingHelixPropagator*>(&*shProp);
	
    lastrecostate = ThisshProp->propagate(startrecostate, *plane);
	
    FreeTrajectoryState finalrecostate;
    lastrecostate.getFreeState(finalrecostate);
      
    AlgebraicSymMatrix66 covFinalReco;
    GlobalVector p3FinalReco_glob, r3FinalReco_globv;
    getFromFTS(finalrecostate, p3FinalReco_glob, r3FinalReco_globv, chargeReco, covFinalReco);
    GlobalPoint r3FinalReco_glob(r3FinalReco_globv.x(),r3FinalReco_globv.y(),r3FinalReco_globv.z());

    double DirectionPull, DirectionPullNum, DirectionPullDenom;

    //Computing the sigma for the track direction
    Double_t mag_track = p3FinalReco_glob.perp();
    //Double_t phi_track = p3FinalReco_glob.phi();

    //Double_t dmagdx_track = p3FinalReco_glob.x()/mag_track;
    //Double_t dmagdy_track = p3FinalReco_glob.y()/mag_track;
    Double_t dphidx_track = -p3FinalReco_glob.y()/(mag_track*mag_track);
    Double_t dphidy_track = p3FinalReco_glob.x()/(mag_track*mag_track);
    Double_t sigmaphi_track = sqrt( dphidx_track*dphidx_track*covFinalReco(3,3)+
    				dphidy_track*dphidy_track*covFinalReco(4,4)+
    				dphidx_track*dphidy_track*2*covFinalReco(3,4) );

    DirectionPullNum = p3FinalReco_glob.phi()-roll->toGlobal(Seg.localDirection()).phi();
    DirectionPullDenom = sqrt( pow(roll->toGlobal(Seg.localPosition()).phi(),2) + pow(sigmaphi_track,2) );
    DirectionPull = DirectionPullNum / DirectionPullDenom;

    if ( (tkRef->pt() > 5.0)&& (IsMatched[MuID]) ){
      SegTrackDirPhiDiff_True_h->Fill(p3FinalReco_glob.phi()-roll->toGlobal(Seg.localDirection()).phi() );
      SegTrackDirEtaDiff_True_h->Fill(p3FinalReco_glob.eta()-roll->toGlobal(Seg.localDirection()).eta() );
      SegTrackDirPhiPull_True_h->Fill(DirectionPull);
    }
    SegTrackDirPhiDiff_All_h->Fill(p3FinalReco_glob.phi()-roll->toGlobal(Seg.localDirection()).phi() );
    SegTrackDirPhiPull_All_h->Fill(DirectionPull);
    
    SegTrackDirEtaDiff_All_h->Fill(p3FinalReco_glob.eta()-roll->toGlobal(Seg.localDirection()).eta() );


    LocalPoint r3FinalReco = roll->toLocal(r3FinalReco_glob);
    LocalVector p3FinalReco=roll->toLocal(p3FinalReco_glob);
    LocalTrajectoryParameters ltp(r3FinalReco,p3FinalReco,chargeReco);
    JacobianCartesianToLocal jctl(roll->surface(),ltp);
    AlgebraicMatrix56 jacobGlbToLoc = jctl.jacobian(); 

    AlgebraicMatrix55 Ctmp =  (jacobGlbToLoc * covFinalReco) * ROOT::Math::Transpose(jacobGlbToLoc); 
    AlgebraicSymMatrix55 C;  // I couldn't find any other way, so I resort to the brute force
    for(int i=0; i<5; ++i) {
      for(int j=0; j<5; ++j) {
	C[i][j] = Ctmp[i][j]; 

      }
    }  

    LocalPoint thisPosition(Seg.localPosition());

    Double_t sigmax = sqrt(C[3][3]+Seg.localPositionError().xx() );      
    Double_t sigmay = sqrt(C[4][4]+Seg.localPositionError().yy() );

    XPull_h->Fill((thisPosition.x()-r3FinalReco.x())/sigmax);
    YPull_h->Fill((thisPosition.y()-r3FinalReco.y())/sigmay);
    
    XDiff_h->Fill((thisPosition.x()-r3FinalReco.x()));
    YDiff_h->Fill((thisPosition.y()-r3FinalReco.y()));

    //std::cout<<"AM HERE"<<std::endl;
    if ( (tkRef->pt() > FakeRatePtCut)&& (IsMatched[MuID]) ){
      


      //std::cout<<"thisPosition = "<<thisPosition<<std::endl;
      //std::cout<<"r3FinalReco = "<<r3FinalReco<<std::endl;
    }

    //End Direction studies


    bool IsNew = true;
    for (unsigned int i =0; i < Ids.size(); i++){
      if (SegId == Ids[i]) IsNew=false;
    }

    if (IsNew) {
      UniqueIdList.push_back(SegId);
      //std::cout<<"New SegId = "<<SegId<<std::endl;
      //std::cout<<GlobVect<<std::endl;
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
      if ( (tkRef->pt() > 5.0) && (tkRef->pt() <= 10.0) )  	ME0Muon_Cuts_Eta_5_10->Fill(fabs(tkRef->eta()));
      if ( (tkRef->pt() > 10.0) && (tkRef->pt() <= 20.0) )	ME0Muon_Cuts_Eta_10_20->Fill(fabs(tkRef->eta()));
      if ( (tkRef->pt() > 20.0) && (tkRef->pt() <= 40.0) )	ME0Muon_Cuts_Eta_20_40->Fill(fabs(tkRef->eta()));
      if ( tkRef->pt() > 40.0) 		ME0Muon_Cuts_Eta_40->Fill(fabs(tkRef->eta()));
    }
    


    if ( (TMath::Abs(tkRef->eta()) > 2.0) && (TMath::Abs(tkRef->eta()) < 2.8) ) ME0Muon_Pt->Fill(tkRef->pt());

    TrackID++;
    MuID++;
  }
  
   std::cout<<UniqueIdList.size()<<" unique segments per event"<<std::endl;
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

    // if (SegmentEta[i] > 2.4){
    //   Segment_Eta->Fill(SegmentEta[i]);
    //   Segment_Phi->Fill(SegmentPhi[i]);
    //   Segment_R->Fill(SegmentR[i]);
    //   Segment_Pos->Fill(SegmentX[i],SegmentY[i]);
    // }
  }

  //================  For Segment Plotting
  for (auto thisSegment = OurSegments->begin(); thisSegment != OurSegments->end(); 
       ++thisSegment){
    ME0DetId id = thisSegment->me0DetId();
    //std::cout<<"ME0DetId =  "<<id<<std::endl;
    auto roll = me0Geom->etaPartition(id); 
    auto GlobVect(roll->toGlobal(thisSegment->localPosition()));
    Segment_Eta->Fill(fabs(GlobVect.eta()));
    Segment_Phi->Fill(GlobVect.phi());
    Segment_R->Fill(GlobVect.perp());
    Segment_Pos->Fill(GlobVect.x(),GlobVect.y());



    auto theseRecHits = thisSegment->specificRecHits();
    //std::cout <<"ME0 Ensemble Det Id "<<id<<"  Number of RecHits "<<theseRecHits.size()<<std::endl;
    
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
	  //std::cout<<"ME0DetId =  "<<id<<std::endl;
	  auto roll = me0Geom->etaPartition(id); 
	  auto GlobVect(roll->toGlobal(thisSegment->localPosition()));
	  Segment_Eta->Fill(fabs(GlobVect.eta()));
	  Segment_Phi->Fill(GlobVect.phi());
	  Segment_R->Fill(GlobVect.perp());
	  Segment_Pos->Fill(GlobVect.x(),GlobVect.y());
	  
	  if (reco::deltaR(CurrentParticle,GlobVect) < SmallestDelR) SmallestDelR = reco::deltaR(CurrentParticle,GlobVect);

	}
	if ((fabs(CurrentParticle.eta()) < 2.0 ) ||(fabs(CurrentParticle.eta()) > 2.8 )) continue;
	DelR_Segment_GenMuon->Fill(SmallestDelR);
      }
    }
  

  //std::cout<<recosize<<std::endl;
  for (std::vector<RecoChargedCandidate>::const_iterator thisCandidate = OurCandidates->begin();
       thisCandidate != OurCandidates->end(); ++thisCandidate){
    TLorentzVector CandidateVector;
    CandidateVector.SetPtEtaPhiM(thisCandidate->pt(),thisCandidate->eta(),thisCandidate->phi(),0);
    //std::cout<<"On a muon"<<std::endl;
    //std::cout<<thisCandidate->eta()<<std::endl;
    Candidate_Eta->Fill(fabs(thisCandidate->eta()));
  }

  if (OurCandidates->size() == 2){
    TLorentzVector CandidateVector1,CandidateVector2;
    CandidateVector1.SetPtEtaPhiM((*OurCandidates)[0].pt(),(*OurCandidates)[0].eta(),(*OurCandidates)[0].phi(),0);
    CandidateVector2.SetPtEtaPhiM((*OurCandidates)[1].pt(),(*OurCandidates)[1].eta(),(*OurCandidates)[1].phi(),0);
    Double_t Mass = (CandidateVector1+CandidateVector2).M();
    Mass_h->Fill(Mass);
  }
  
  
}

void ME0MuonAnalyzer::endJob() 
{

  std::cout<<"Nevents = "<<Nevents<<std::endl;
  //TString cmsText     = "CMS Prelim.";
  //TString cmsText     = "#splitline{CMS PhaseII Simulation}{Prelim}";
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
  Segment_Eta->GetYaxis()->SetTitle(" \# of Segments");
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
  //c1->SetLogy();
  ME0Muon_Pt->Write();   ME0Muon_Pt->Draw();  
  ME0Muon_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  ME0Muon_Pt->GetXaxis()->SetTitleSize(0.05);
  c1->Print(histoFolder+"/ME0Muon_Pt.png");

  GenMuon_Eta->Write();   GenMuon_Eta->Draw();  c1->Print(histoFolder+"/GenMuon_Eta.png");

  GenMuon_Pt->Write();   GenMuon_Pt->Draw();  c1->Print(histoFolder+"/GenMuon_Pt.png");

  MatchedME0Muon_Eta->Write();   MatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_Eta.png");

  Chi2MatchedME0Muon_Eta->Write();   Chi2MatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/Chi2MatchedME0Muon_Eta.png");
  Chi2UnmatchedME0Muon_Eta->Write();   Chi2UnmatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/Chi2UnmatchedME0Muon_Eta.png");

  gStyle->SetOptStat(1);
  MatchedME0Muon_Pt->GetXaxis()->SetTitle("ME0Muon p_{T}");
  //MatchedME0Muon_Pt->GetYaxis()->SetTitle(" \# of Se");

  MatchedME0Muon_Pt->Write();   MatchedME0Muon_Pt->Draw();  c1->Print(histoFolder+"/MatchedME0Muon_Pt.png");
  gStyle->SetOptStat(0);

  UnmatchedME0Muon_Eta->Write();   UnmatchedME0Muon_Eta->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Eta.png");
  UnmatchedME0Muon_Cuts_Eta->Write();   UnmatchedME0Muon_Cuts_Eta->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Cuts_Eta.png");
  //gStyle->SetOptStat('oue');
  c1->SetLogy();
  UnmatchedME0Muon_Pt->Write();   UnmatchedME0Muon_Pt->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Pt.png");
  gStyle->SetOptStat(0);
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
  
  UnmatchedME0Muon_Cuts_Eta_PerEvent->GetYaxis()->SetTitle("Average \# ME0Muons per event");
  UnmatchedME0Muon_Cuts_Eta_PerEvent->GetYaxis()->SetTitleSize(0.05);

  UnmatchedME0Muon_Cuts_Eta_PerEvent->Write();   UnmatchedME0Muon_Cuts_Eta_PerEvent->Draw();  c1->Print(histoFolder+"/UnmatchedME0Muon_Cuts_Eta_PerEvent.png");

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

  GenMuon_Eta->Sumw2();  MatchedME0Muon_Eta->Sumw2();  Chi2MatchedME0Muon_Eta->Sumw2();   Chi2UnmatchedME0Muon_Eta->Sumw2();
  GenMuon_Pt->Sumw2();  MatchedME0Muon_Pt->Sumw2();

  Track_Eta->Sumw2();  ME0Muon_Eta->Sumw2();
  Track_Pt->Sumw2();  ME0Muon_Pt->Sumw2();

  UnmatchedME0Muon_Eta->Sumw2();
  UnmatchedME0Muon_Pt->Sumw2();
  
  UnmatchedME0Muon_Cuts_Eta->Sumw2();    ME0Muon_Cuts_Eta->Sumw2();

  ME0Muon_Cuts_Eta_5_10->Sumw2();  ME0Muon_Cuts_Eta_10_20->Sumw2();  ME0Muon_Cuts_Eta_20_40->Sumw2();  ME0Muon_Cuts_Eta_40->Sumw2();
  UnmatchedME0Muon_Cuts_Eta_5_10->Sumw2();  UnmatchedME0Muon_Cuts_Eta_10_20->Sumw2();  UnmatchedME0Muon_Cuts_Eta_20_40->Sumw2();  UnmatchedME0Muon_Cuts_Eta_40->Sumw2();
  GenMuon_Eta_5_10->Sumw2();  GenMuon_Eta_10_20->Sumw2();  GenMuon_Eta_20_40->Sumw2();  GenMuon_Eta_40->Sumw2();
  MatchedME0Muon_Eta_5_10->Sumw2();  MatchedME0Muon_Eta_10_20->Sumw2();  MatchedME0Muon_Eta_20_40->Sumw2();  MatchedME0Muon_Eta_40->Sumw2();
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

  MuonRecoEff_Eta_10_20->Divide(MatchedME0Muon_Eta_10_20, GenMuon_Eta_10_20, 1, 1, "B");
  MuonRecoEff_Eta_10_20->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_10_20->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_10_20->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_10_20->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_10_20->SetMinimum(MuonRecoEff_Eta_10_20->GetMinimum()-0.1);
  MuonRecoEff_Eta_10_20->SetMinimum(0);
  //MuonRecoEff_Eta_10_20->SetMaximum(MuonRecoEff_Eta_10_20->GetMaximum()+0.1);
  MuonRecoEff_Eta_10_20->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_10_20->Write();   MuonRecoEff_Eta_10_20->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_10_20.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_10_20.png");


  MuonRecoEff_Eta_20_40->Divide(MatchedME0Muon_Eta_20_40, GenMuon_Eta_20_40, 1, 1, "B");
  MuonRecoEff_Eta_20_40->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_20_40->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_20_40->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_20_40->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_20_40->SetMinimum(MuonRecoEff_Eta_20_40->GetMinimum()-0.1);
  MuonRecoEff_Eta_20_40->SetMinimum(0);
  //MuonRecoEff_Eta_20_40->SetMaximum(MuonRecoEff_Eta_20_40->GetMaximum()+0.1);
  MuonRecoEff_Eta_20_40->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_20_40->Write();   MuonRecoEff_Eta_20_40->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_20_40.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_20_40.png");


  MuonRecoEff_Eta_40->Divide(MatchedME0Muon_Eta_40, GenMuon_Eta_40, 1, 1, "B");
  MuonRecoEff_Eta_40->GetXaxis()->SetTitle("Gen Muon |#eta|");
  MuonRecoEff_Eta_40->GetXaxis()->SetTitleSize(0.05);
  MuonRecoEff_Eta_40->GetYaxis()->SetTitle("ME0Muon Efficiency");
  MuonRecoEff_Eta_40->GetYaxis()->SetTitleSize(0.05);
  //MuonRecoEff_Eta_40->SetMinimum(MuonRecoEff_Eta_40->GetMinimum()-0.1);
  MuonRecoEff_Eta_40->SetMinimum(0);
  //MuonRecoEff_Eta_40->SetMaximum(MuonRecoEff_Eta_40->GetMaximum()+0.1);
  MuonRecoEff_Eta_40->SetMaximum(1.2);
  //CMS_lumi( c1, 7, 11 );
  MuonRecoEff_Eta_40->Write();   MuonRecoEff_Eta_40->Draw();  
  txt->DrawLatex(0.15,0.2,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);

  //c1->SaveAs("TestMuonRecoEff_Eta_40.png");
  c1->Print(histoFolder+"/MuonRecoEff_Eta_40.png");


  Chi2MuonRecoEff_Eta->Divide(Chi2MatchedME0Muon_Eta, GenMuon_Eta, 1, 1, "B");
  std::cout<<"GenMuon_Eta =  "<<GenMuon_Eta->Integral()<<std::endl;
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



  FakeRate_Eta_10_20->Divide(UnmatchedME0Muon_Cuts_Eta_10_20, ME0Muon_Cuts_Eta_10_20, 1, 1, "B");
  FakeRate_Eta_10_20->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_10_20->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_10_20->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_10_20->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_10_20->SetMinimum(FakeRate_Eta_10_20->GetMinimum()-0.1);
  FakeRate_Eta_10_20->SetMinimum(0);
  //FakeRate_Eta_10_20->SetMaximum(FakeRate_Eta_10_20->GetMaximum()+0.1);
  FakeRate_Eta_10_20->SetMaximum(1.2);
  FakeRate_Eta_10_20->Write();   FakeRate_Eta_10_20->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_10_20.png');
  c1->Print(histoFolder+"/FakeRate_Eta_10_20.png");



  FakeRate_Eta_20_40->Divide(UnmatchedME0Muon_Cuts_Eta_20_40, ME0Muon_Cuts_Eta_20_40, 1, 1, "B");
  FakeRate_Eta_20_40->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_20_40->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_20_40->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_20_40->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_20_40->SetMinimum(FakeRate_Eta_20_40->GetMinimum()-0.1);
  FakeRate_Eta_20_40->SetMinimum(0);
  //FakeRate_Eta_20_40->SetMaximum(FakeRate_Eta_20_40->GetMaximum()+0.1);
  FakeRate_Eta_20_40->SetMaximum(1.2);
  FakeRate_Eta_20_40->Write();   FakeRate_Eta_20_40->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_20_40.png');
  c1->Print(histoFolder+"/FakeRate_Eta_20_40.png");



  FakeRate_Eta_40->Divide(UnmatchedME0Muon_Cuts_Eta_40, ME0Muon_Cuts_Eta_40, 1, 1, "B");
  FakeRate_Eta_40->GetXaxis()->SetTitle("Reconstructed track |#eta|");
  FakeRate_Eta_40->GetXaxis()->SetTitleSize(0.05);
  FakeRate_Eta_40->GetYaxis()->SetTitle("ME0 Muon Fake Rate");
  FakeRate_Eta_40->GetYaxis()->SetTitleSize(0.05);
  //FakeRate_Eta_40->SetMinimum(FakeRate_Eta_40->GetMinimum()-0.1);
  FakeRate_Eta_40->SetMinimum(0);
  //FakeRate_Eta_40->SetMaximum(FakeRate_Eta_40->GetMaximum()+0.1);
  FakeRate_Eta_40->SetMaximum(1.2);
  FakeRate_Eta_40->Write();   FakeRate_Eta_40->Draw();  

  txt->DrawLatex(0.15,0.4,pcstr);
  latex->DrawLatex(0.4, 0.85, cmsText);
  latex1->DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

  //c1->SaveAs('TestFakeRate_Eta_40.png');
  c1->Print(histoFolder+"/FakeRate_Eta_40.png");



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
    test= new TH1D("test"   , "pt resolution"   , 200, -1.0, 1.0 );  
  
    for(Int_t i=1; i<=PtDiff_s->GetNbinsX(); ++i) {
    
    std::stringstream tempstore;
    tempstore<<i;
    const std::string& thistemp = tempstore.str();
    test->Draw();


    PtDiff_s->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    // TF1 *gaus_narrow = new TF1("gaus_narrow","gaus",-.1,.1);
    // test->Fit(gaus_narrow,"R");

    // Double_t n0  = gaus_narrow->GetParameter(0);
    // Double_t n1  = gaus_narrow->GetParameter(1);
    // Double_t n2  = gaus_narrow->GetParameter(2);

    // //Double_t e_n0  = gaus_narrow->GetParameterError(0);
    // //Double_t e_n1  = gaus_narrow->GetParameterError(1);
    // Double_t e_n2  = gaus_narrow->GetParError(2);

    // std::cout<<n0<<", "<<n1<<", "<<n2<<std::endl;

    TF1 *gaus_wide = new TF1("gaus_wide","gaus",-.2,.2);
    test->Fit(gaus_wide,"R");

    Double_t w2  = gaus_wide->GetParameter(2);

    Double_t e_w2  = gaus_wide->GetParError(2);

    // PtDiff_gaus_narrow->SetBinContent(i, n2); 
    // PtDiff_gaus_narrow->SetBinError(i, e_n2); 
    PtDiff_gaus_wide->SetBinContent(i, w2); 
    PtDiff_gaus_wide->SetBinError(i, e_w2); 
    
    test->Write();
    TString FileName = "Bin"+thistemp+"Fit.png";
    c1->Print(histoFolder+"/"+FileName);

    test->Draw();

    delete test;
    // Redoing for pt 5 to 10
    PtDiff_s_5_10->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

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
    // Redoing for pt 10 to 20
    PtDiff_s_10_20->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *gaus_10_20 = new TF1("gaus_10_20","gaus",-.2,.2);
    test->Fit(gaus_10_20,"R");

     w2  = gaus_10_20->GetParameter(2);
     e_w2  = gaus_10_20->GetParError(2);

    PtDiff_gaus_10_20->SetBinContent(i, w2); 
    PtDiff_gaus_10_20->SetBinError(i, e_w2); 

    test->Draw();
    FileName = "Bin"+thistemp+"Fit_10_20.png";
    c1->Print(histoFolder+"/"+FileName);

    delete test;
    // Redoing for pt 20 to 40
    PtDiff_s_20_40->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *gaus_20_40 = new TF1("gaus_20_40","gaus",-.2,.2);
    test->Fit(gaus_20_40,"R");

     w2  = gaus_20_40->GetParameter(2);
     e_w2  = gaus_20_40->GetParError(2);

    PtDiff_gaus_20_40->SetBinContent(i, w2); 
    PtDiff_gaus_20_40->SetBinError(i, e_w2); 

    test->Draw();
    FileName = "Bin"+thistemp+"Fit_20_40.png";
    c1->Print(histoFolder+"/"+FileName);

    delete test;
    // Redoing for pt 40+
    PtDiff_s_40->ProjectionY("test",i,i,"");
    if (test->Integral() < 1.0) continue;

    TF1 *gaus_40 = new TF1("gaus_40","gaus",-.2,.2);
    test->Fit(gaus_40,"R");

     w2  = gaus_40->GetParameter(2);
     e_w2  = gaus_40->GetParError(2);

    PtDiff_gaus_40->SetBinContent(i, w2); 
    PtDiff_gaus_40->SetBinError(i, e_w2); 

    test->Draw();
    FileName = "Bin"+thistemp+"Fit_40.png";
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

  PtDiff_gaus_10_20->SetMarkerStyle(22); 
  PtDiff_gaus_10_20->SetMarkerSize(1.2); 
  PtDiff_gaus_10_20->SetMarkerColor(kBlue); 
  //PtDiff_gaus_10_20->SetLineColor(kRed); 
  
  //PtDiff_gaus_10_20->Draw("PL"); 

  PtDiff_gaus_10_20->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_10_20->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_10_20->Write();     PtDiff_gaus_10_20->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus_10_20.png");

  PtDiff_gaus_20_40->SetMarkerStyle(22); 
  PtDiff_gaus_20_40->SetMarkerSize(1.2); 
  PtDiff_gaus_20_40->SetMarkerColor(kBlue); 
  //PtDiff_gaus_20_40->SetLineColor(kRed); 
  
  //PtDiff_gaus_20_40->Draw("PL"); 

  PtDiff_gaus_20_40->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_20_40->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_20_40->Write();     PtDiff_gaus_20_40->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus_20_40.png");

  PtDiff_gaus_40->SetMarkerStyle(22); 
  PtDiff_gaus_40->SetMarkerSize(1.2); 
  PtDiff_gaus_40->SetMarkerColor(kBlue); 
  //PtDiff_gaus_40->SetLineColor(kRed); 
  
  //PtDiff_gaus_40->Draw("PL"); 

  PtDiff_gaus_40->GetXaxis()->SetTitle("Gen Muon |#eta|");
  PtDiff_gaus_40->GetYaxis()->SetTitle("Gaussian width of (pt track-ptgen)/ptgen");
  PtDiff_gaus_40->Write();     PtDiff_gaus_40->Draw("PE");  c1->Print(histoFolder+"/PtDiff_gaus_40.png");


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

  logout<<"Fake Rate 5_10:\n";
  for (int i=1; i<=FakeRate_Eta_5_10->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_5_10->GetBinContent(i)<<","<<FakeRate_Eta_5_10->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 5_10:\n";
  for (int i=1; i<=PtDiff_gaus_5_10->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_5_10->GetBinContent(i)<<","<<PtDiff_gaus_5_10->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 10_20:\n";
  for (int i=1; i<=MuonRecoEff_Eta_10_20->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_10_20->GetBinContent(i)<<","<<MuonRecoEff_Eta_10_20->GetBinError(i)<<"\n";
  }    

  logout<<"Fake Rate 10_20:\n";
  for (int i=1; i<=FakeRate_Eta_10_20->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_10_20->GetBinContent(i)<<","<<FakeRate_Eta_10_20->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 10_20:\n";
  for (int i=1; i<=PtDiff_gaus_10_20->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_10_20->GetBinContent(i)<<","<<PtDiff_gaus_10_20->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 20_40:\n";
  for (int i=1; i<=MuonRecoEff_Eta_20_40->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_20_40->GetBinContent(i)<<","<<MuonRecoEff_Eta_20_40->GetBinError(i)<<"\n";
  }    

  logout<<"Fake Rate 20_40:\n";
  for (int i=1; i<=FakeRate_Eta_20_40->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_20_40->GetBinContent(i)<<","<<FakeRate_Eta_20_40->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 20_40:\n";
  for (int i=1; i<=PtDiff_gaus_20_40->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_20_40->GetBinContent(i)<<","<<PtDiff_gaus_20_40->GetBinError(i)<<"\n";
  }    


  logout<<"Efficiencies and errors 40:\n";
  for (int i=1; i<=MuonRecoEff_Eta_40->GetNbinsX(); ++i){
    logout<<MuonRecoEff_Eta_40->GetBinContent(i)<<","<<MuonRecoEff_Eta_40->GetBinError(i)<<"\n";
  }    

  logout<<"Fake Rate 40:\n";
  for (int i=1; i<=FakeRate_Eta_40->GetNbinsX(); ++i){
    logout<<FakeRate_Eta_40->GetBinContent(i)<<","<<FakeRate_Eta_40->GetBinError(i)<<"\n";
  }    

  logout<<"Resolution vs eta 40:\n";
  for (int i=1; i<=PtDiff_gaus_40->GetNbinsX(); ++i){
    logout<<PtDiff_gaus_40->GetBinContent(i)<<","<<PtDiff_gaus_40->GetBinError(i)<<"\n";
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
