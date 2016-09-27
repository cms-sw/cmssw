#ifndef RecoParticleFlow_PFProducer_PFMuonAlgo_h
#define RecoParticleFlow_PFProducer_PFMuonAlgo_h 
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class PFMuonAlgo {

  typedef reco::Muon::MuonTrackTypePair MuonTrackTypePair;
  typedef reco::Muon::MuonTrackType  MuonTrackType;

 public:

  /// constructor
  PFMuonAlgo();

  void setParameters(const edm::ParameterSet&);
  


  /// destructor
  virtual ~PFMuonAlgo() {;}
  
  ////STATIC MUON ID METHODS

  static bool isMuon( const reco::PFBlockElement& elt );

  static bool isLooseMuon( const reco::PFBlockElement& elt );

  static bool isGlobalTightMuon( const reco::PFBlockElement& elt );

  static bool isGlobalLooseMuon( const reco::PFBlockElement& elt );

  static bool isTrackerTightMuon( const reco::PFBlockElement& elt );

  static bool isTrackerLooseMuon( const reco::PFBlockElement& elt );

  static bool isIsolatedMuon( const reco::PFBlockElement& elt );

  static bool isMuon( const reco::MuonRef& muonRef );  

  static bool isLooseMuon( const reco::MuonRef& muonRef );

  static bool isGlobalTightMuon( const reco::MuonRef& muonRef );

  static bool isGlobalLooseMuon( const reco::MuonRef& muonRef );




  static bool isTrackerTightMuon( const reco::MuonRef& muonRef );
  
  static bool isTrackerLooseMuon( const reco::MuonRef& muonRef );
  
  static bool isIsolatedMuon( const reco::MuonRef& muonRef );

  static bool isTightMuonPOG(const reco::MuonRef& muonRef);

  static void printMuonProperties( const reco::MuonRef& muonRef );




  ////POST CLEANING AND MOMEMNTUM ASSIGNMENT
  bool hasValidTrack(const reco::MuonRef& muonRef,bool loose =false);




  //Make a PF Muon : Basic method
  bool reconstructMuon(reco::PFCandidate&, const reco::MuonRef&,bool allowLoose = false);


  //Assign a different track to the muon
  void changeTrack(reco::PFCandidate&,const MuonTrackTypePair&);
  //PF Post cleaning algorithm
  void setInputsForCleaning(const reco::VertexCollection*); 
  void postClean(reco::PFCandidateCollection *);
  void addMissingMuons(edm::Handle<reco::MuonCollection>, reco::PFCandidateCollection* cands);

  std::unique_ptr<reco::PFCandidateCollection> transferCleanedCosmicCandidates() {
    return std::move(pfCosmicsMuonCleanedCandidates_);
  }

  std::unique_ptr<reco::PFCandidateCollection> transferCleanedTrackerAndGlobalCandidates() {
    return std::move(pfCleanedTrackerAndGlobalMuonCandidates_);
  }

  std::unique_ptr<reco::PFCandidateCollection> transferCleanedFakeCandidates() {
    return std::move(pfFakeMuonCleanedCandidates_);
  }

  std::unique_ptr<reco::PFCandidateCollection> transferPunchThroughCleanedMuonCandidates() {
    return std::move(pfPunchThroughMuonCleanedCandidates_);
  }

  std::unique_ptr<reco::PFCandidateCollection> transferPunchThroughCleanedHadronCandidates() {
    return std::move(pfPunchThroughHadronCleanedCandidates_);
  }

  std::unique_ptr<reco::PFCandidateCollection> transferAddedMuonCandidates() {
    return std::move(pfAddedMuonCandidates_);
  }

 private:
  //Gives the track with the smallest Dpt/Pt
  MuonTrackTypePair getTrackWithSmallestError(const std::vector<MuonTrackTypePair>&);

  std::vector<reco::Muon::MuonTrackTypePair> muonTracks(const reco::MuonRef& muon,bool includeSA = false,double dpt = 1e+9);

  //Gets the good tracks
  std::vector<reco::Muon::MuonTrackTypePair> goodMuonTracks(const reco::MuonRef& muon,bool includeSA = false);


  //Estimate MET and SUmET for post cleaning
  void estimateEventQuantities(const reco::PFCandidateCollection*  );

  //Post cleaning Sub-methods
  bool cleanMismeasured(reco::PFCandidate&,unsigned int);
  bool cleanPunchThroughAndFakes(reco::PFCandidate&,reco::PFCandidateCollection* ,unsigned int );

  void  removeDeadCandidates(reco::PFCandidateCollection*, const std::vector<unsigned int>&);



  //helpers  
  std::pair<double,double> getMinMaxMET2(const reco::PFCandidate&);
  std::vector<MuonTrackTypePair> tracksWithBetterMET(const std::vector<MuonTrackTypePair>& ,const reco::PFCandidate&);
  std::vector<MuonTrackTypePair> tracksPointingAtMET(const std::vector<MuonTrackTypePair>&);

  //Output collections for post cleaning
  /// the collection of  cosmics cleaned muon candidates
  std::unique_ptr<reco::PFCandidateCollection> pfCosmicsMuonCleanedCandidates_;
  /// the collection of  tracker/global cleaned muon candidates
  std::unique_ptr<reco::PFCandidateCollection> pfCleanedTrackerAndGlobalMuonCandidates_;
  /// the collection of  fake cleaned muon candidates
  std::unique_ptr<reco::PFCandidateCollection> pfFakeMuonCleanedCandidates_;
  /// the collection of  punch-through cleaned muon candidates
  std::unique_ptr<reco::PFCandidateCollection> pfPunchThroughMuonCleanedCandidates_;
  /// the collection of  punch-through cleaned neutral hadron candidates
  std::unique_ptr<reco::PFCandidateCollection> pfPunchThroughHadronCleanedCandidates_;
  /// the collection of  added muon candidates
  std::unique_ptr<reco::PFCandidateCollection> pfAddedMuonCandidates_;
  
  std::vector<unsigned int > maskedIndices_;

  
  //////////////////////////////////////////////////////////////////////////////////////
  const reco::VertexCollection *  vertices_;



  //Configurables
  double maxDPtOPt_;
  int minTrackerHits_;
  int minPixelHits_;
  reco::TrackBase::TrackQuality trackQuality_;
  
  double errorCompScale_;
  double eventFractionCleaning_;
  double sumetPU_;
  double dzPV_;
  bool postCleaning_;
  double minPostCleaningPt_;
  double eventFactorCosmics_;
  double metSigForCleaning_;
  double metSigForRejection_;
  double metFactorCleaning_;
  double eventFractionRejection_;
  double metFactorRejection_;
  double metFactorHighEta_;
  double ptFactorHighEta_;
  double metFactorFake_;
  double minPunchThroughMomentum_;
  double minPunchThroughEnergy_;
  double punchThroughFactor_;
  double punchThroughMETFactor_;
  double cosmicRejDistance_;

  double sumet_;
  double METX_;
  double METY_;


  ///////COMPARATORS

  class TrackMETComparator {
  public:
    TrackMETComparator(double METX,double METY) {metx_ = METX; mety_=METY;}
    ~TrackMETComparator() {}
    
    bool operator()(const MuonTrackTypePair& mu1,const MuonTrackTypePair& mu2) {
      return pow(metx_+mu1.first->px(),2)+pow(mety_+mu1.first->py(),2) < pow(metx_+mu2.first->px(),2)+pow(mety_+mu2.first->py(),2);
    }
  private:
    double metx_;
    double mety_;


  };


  class IndexPtComparator {
  public:

    IndexPtComparator(const reco::PFCandidateCollection* coll):coll_(coll) {
    }
    ~IndexPtComparator() {}
    
    bool operator()(int mu1,int mu2) {
      return coll_->at(mu1).pt() > coll_->at(mu2).pt();
    }

  private:
    const reco::PFCandidateCollection * coll_;

  };




  class TrackPtErrorSorter {
  public:
    TrackPtErrorSorter() {}
    ~TrackPtErrorSorter() {}

    bool operator()(const MuonTrackTypePair& mu1,const MuonTrackTypePair& mu2) {
      return mu1.first->ptError()/mu1.first->pt() < mu2.first->ptError()/mu2.first->pt();
    }
  };



};

#endif
