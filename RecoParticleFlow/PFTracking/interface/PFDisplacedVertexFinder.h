#ifndef RecoParticleFlow_PFTracking_PFDisplacedVertexFinder_h
#define RecoParticleFlow_PFTracking_PFDisplacedVertexFinder_h 

#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexHelper.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexSeed.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexSeedFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoParticleFlow/PFTracking/interface/PFCheckHitPattern.h"


/// \brief Displaced Vertex Finder Algorithm
/*!
  \author Maxime Gouzevitch
  \date October 2009
*/

class TrackingGeometry;
class TrackerGeometry;
class MagneticField;

class PFDisplacedVertexFinder {

 public:

  PFDisplacedVertexFinder();

  ~PFDisplacedVertexFinder();

  /// -------- Useful Types -------- ///

  typedef reco::PFDisplacedVertexSeedCollection::iterator IDVS;
  typedef reco::PFDisplacedVertexCollection::iterator IDV;

  typedef std::pair <unsigned int, unsigned int> PFTrackHitInfo;
  typedef std::pair <PFTrackHitInfo, PFTrackHitInfo> PFTrackHitFullInfo;

  /// Fitter Type
  enum FitterType {
    F_NOTDEFINED,
    F_DONOTREFIT,
    F_KALMAN,
    F_ADAPTIVE
  };


  /// -------- Set different algo parameters -------- ///

  /// Sets algo parameters for the vertex finder
  void setParameters(double transvSize, double longSize, 
		     double primaryVertexCut, double tobCut, 
		     double tecCut, double minAdaptWeight, bool switchOff2TrackVertex) {
    transvSize_ = transvSize;
    longSize_   = longSize;
    primaryVertexCut_ = primaryVertexCut;
    tobCut_ = tobCut;
    tecCut_ = tecCut;
    minAdaptWeight_ = minAdaptWeight;
    switchOff2TrackVertex_ = switchOff2TrackVertex;
  }

  /// Sets debug printout flag
  void setDebug( bool debug ) {debug_ = debug;}

  /// Sets parameters for track extrapolation and hits study
  void setEdmParameters( const MagneticField* magField,
			 edm::ESHandle<GlobalTrackingGeometry> globTkGeomHandle,
			 const TrackerTopology* tkerTopo,
			 const TrackerGeometry* tkerGeom){
    magField_ = magField; 
    globTkGeomHandle_ = globTkGeomHandle;
    tkerTopo_ = tkerTopo;
    tkerGeom_ = tkerGeom;
  }

  void setTracksSelector(const edm::ParameterSet& ps){
    helper_.setTracksSelector(ps);
  }

  void setVertexIdentifier(const edm::ParameterSet& ps){
    helper_.setVertexIdentifier(ps);
  }

  void setPrimaryVertex(edm::Handle< reco::VertexCollection > mainVertexHandle, 
			edm::Handle< reco::BeamSpot > beamSpotHandle){
    helper_.setPrimaryVertex(mainVertexHandle, beamSpotHandle);
  }

  void setAVFParameters(const edm::ParameterSet& ps){
      sigmacut_ = ps.getParameter<double>("sigmacut");
      t_ini_    = ps.getParameter<double>("Tini");
      ratio_    = ps.getParameter<double>("ratio");
  }

  /// Set input collections of tracks
  void  setInput(const edm::Handle< reco::PFDisplacedVertexCandidateCollection >&); 
  
  
  /// \return unique_ptr to collection of DisplacedVertices
  std::unique_ptr< reco::PFDisplacedVertexCollection > transferDisplacedVertices() {return std::move(displacedVertices_);}

  const std::unique_ptr< reco::PFDisplacedVertexCollection >& displacedVertices() const {return std::move(displacedVertices_);}



  /// -------- Main function which find vertices -------- ///

  void findDisplacedVertices();


 private:
  
  /// -------- Different steps of the finder algorithm -------- ///

  /// Analyse a vertex candidate and select potential vertex point(s)
  void findSeedsFromCandidate(const reco::PFDisplacedVertexCandidate&, reco::PFDisplacedVertexSeedCollection&);

  /// Sometimes two vertex candidates can be quite close and coming from the same vertex
  void mergeSeeds(reco::PFDisplacedVertexSeedCollection&, std::vector<bool>& bLocked);

  /// Fit one by one the vertex points with associated tracks to get displaced vertices
  bool fitVertexFromSeed(const reco::PFDisplacedVertexSeed&, reco::PFDisplacedVertex&);

  /// Remove potentially fakes displaced vertices
  void selectAndLabelVertices(reco::PFDisplacedVertexCollection&,  std::vector <bool>&);

  bool rejectAndLabelVertex(reco::PFDisplacedVertex& dv);

  /// -------- Tools -------- ///

  bool isCloseTo(const reco::PFDisplacedVertexSeed&, const reco::PFDisplacedVertexSeed&) const;

  std::pair<float,float> getTransvLongDiff(const GlobalPoint&, const GlobalPoint&) const;

  reco::PFDisplacedVertex::VertexTrackType getVertexTrackType(PFTrackHitFullInfo&) const;

  unsigned commonTracks(const reco::PFDisplacedVertex&, const reco::PFDisplacedVertex&) const;

  friend std::ostream& operator<<(std::ostream&, const PFDisplacedVertexFinder&);


  /// -------- Members -------- ///

  reco::PFDisplacedVertexCandidateCollection const*  displacedVertexCandidates_;
  std::unique_ptr< reco::PFDisplacedVertexCollection >    displacedVertices_;

  /// -------- Parameters -------- ///

  /// Algo parameters for the vertex finder

  float transvSize_;
  float longSize_;
  double primaryVertexCut_;
  double tobCut_;
  double tecCut_;
  double minAdaptWeight_;

  bool switchOff2TrackVertex_;

  /// Adaptive Vertex Fitter parameters
  
  double sigmacut_; //= 6;
  double t_ini_; //= 256.;
  double ratio_; //= 0.25;


  /// If true, debug printouts activated
  bool   debug_;
  
  /// Tracker geometry for discerning hit positions
  edm::ESHandle<GlobalTrackingGeometry> globTkGeomHandle_;

  /// doc? 
  const TrackerTopology* tkerTopo_;
  const TrackerGeometry* tkerGeom_;

  /// to be able to extrapolate tracks f
  const MagneticField* magField_;

  
  PFCheckHitPattern hitPattern_;

  PFDisplacedVertexHelper helper_;

};

#endif


