#ifndef RecoParticleFlow_PFTracking_PFDisplacedVertexCandidateFinder_h
#define RecoParticleFlow_PFTracking_PFDisplacedVertexCandidateFinder_h 

#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidateFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

/// \brief Displaced Vertex Candidate Finder
/*!
  \author Maxime Gouzevitch
  \date October 2009
*/

class MagneticField;

class PFDisplacedVertexCandidateFinder {

 public:

  PFDisplacedVertexCandidateFinder();

  ~PFDisplacedVertexCandidateFinder();
  

  /// Mask used to spot if a track is free or not
  typedef std::vector<bool> Mask;

  typedef std::list< reco::TrackBaseRef >::iterator IE;
  typedef std::list< reco::TrackBaseRef >::const_iterator IEC;  
  typedef reco::PFDisplacedVertexCandidateCollection::const_iterator IBC;
  

  /// --------- Set different algo parameters ------ ///

  /// Sets algo parameters for the vertex candidate finder
  void setParameters(double dcaCut, double primaryVertexCut, double dcaPInnerHitCut,
		     const edm::ParameterSet& ps_trk) {
    dcaCut_ = dcaCut;
    primaryVertexCut2_ = primaryVertexCut*primaryVertexCut;
    dcaPInnerHitCut2_ = dcaPInnerHitCut*dcaPInnerHitCut;
    nChi2_max_ = ps_trk.getParameter<double>("nChi2_max");
    pt_min_ = ps_trk.getParameter<double>("pt_min");
    pt_min_prim_ = ps_trk.getParameter<double>("pt_min_prim");
    dxy_ = ps_trk.getParameter<double>("dxy");
  }

  /// sets debug printout flag
  void setDebug( bool debug ) {debug_ = debug;}

  /// Set the imput collection of tracks and calculate their
  /// trajectory parameters the Global Trajectory Parameters
  void setInput(const edm::Handle<reco::TrackCollection>& trackh,  
		const MagneticField* magField ); 
  
  
  /// \return auto_ptr to collection of DisplacedVertexCandidates
  std::unique_ptr< reco::PFDisplacedVertexCandidateCollection > transferVertexCandidates() {return std::move(vertexCandidates_);}

  const std::unique_ptr< reco::PFDisplacedVertexCandidateCollection >& vertexCandidates() const 
    {return std::move(vertexCandidates_);}

  /// -------- Main function which find vertices -------- ///

  void findDisplacedVertexCandidates();


  void setPrimaryVertex(edm::Handle< reco::VertexCollection > mainVertexHandle, 
			edm::Handle< reco::BeamSpot > beamSpotHandle);

 private:

  /// -------- Different steps of the finder algorithm -------- ///

  /// Recursive procedure to associate tracks together
  IE associate(IE next, IE last, reco::PFDisplacedVertexCandidate& tempVertexCandidate);

  /// Check whether 2 elements are linked and fill the link parameters
  void link( const reco::TrackBaseRef& el1, 
	     const reco::TrackBaseRef& el2,
	     double& dist,
	     GlobalPoint& P,
	     reco::PFDisplacedVertexCandidate::VertexLinkTest& linktest);

  /// Compute missing links in the displacedVertexCandidates 
  /// (the recursive procedure does not build all links)  
  void packLinks( reco::PFDisplacedVertexCandidate& vertexCandidate); 


  /// -------- TOOLS -------- //

  /// Allows to calculate the helix aproximation for a given track
  /// which may be then extrapolated to any point.
  GlobalTrajectoryParameters
    getGlobalTrajectoryParameters(const reco::Track*) const;


  /// Quality Criterion on the Pt resolution to select a Track
  bool goodPtResolution( const reco::TrackBaseRef& trackref) const;

  /// A function which gather the information 
  /// if a track is available for vertexing 
  bool isSelected(const reco::TrackBaseRef& trackref)
    { return goodPtResolution(trackref);}

  friend std::ostream& operator<<(std::ostream&, const PFDisplacedVertexCandidateFinder&);
		 


  /// -------- Members -------- ///

  std::unique_ptr< reco::PFDisplacedVertexCandidateCollection > vertexCandidates_;
  

  /// The track refs
  std::list< reco::TrackBaseRef >  eventTracks_;


  /// The trackMask allows to keep the information on the
  /// tracks which are still free and those which are already
  /// used or disabled.
  Mask     trackMask_;
  /// The Trajectories vector allow to calculate snd to store 
  /// only once the track trajectory parameters
  std::vector < GlobalTrajectoryParameters > eventTrackTrajectories_;

  /// ----- Algo parameters for the vertex finder ---- ///

  /// Distance of minimal approach below which 
  /// two tracks are considered as linked together
  double dcaCut_;
  /// Do not reconstruct vertices wich are too close to the beam pipe
  double primaryVertexCut2_;
  /// Maximum distance between the DCA Point and the inner hit of the track
  double dcaPInnerHitCut2_;

  /// Tracks preselection to reduce the combinatorics in PFDisplacedVertexCandidates
  /// this cuts are repeated then in a smarter way in the PFDisplacedVertexFinder
  /// be sure you are consistent between them
  double nChi2_max_;
  double pt_min_;

  double pt_min_prim_;
  double dxy_;

  /// Max number of expected vertexCandidates in the event
  /// Used to allocate the memory and avoid multiple copy
  unsigned vertexCandidatesSize_;
  
  // Two track minimum distance algo
  TwoTrackMinimumDistance theMinimum_;

  math::XYZPoint pvtx_;

  /// if true, debug printouts activated
  bool   debug_;
  
  // Tracker geometry for extrapolation
  const MagneticField* magField_;

};

#endif


