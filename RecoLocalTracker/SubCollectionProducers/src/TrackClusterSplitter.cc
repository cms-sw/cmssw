#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

//added for my stuff H.S.
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

// gavril: the template header files
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate2D.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateSplit.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateDefs.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplate.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplateSplit.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplateDefs.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplateReco.h"

#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

// for strip sim splitting
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <boost/range.hpp>
#include <boost/foreach.hpp>
#include "boost/multi_array.hpp"

#include <iostream>
using namespace std;

class TrackClusterSplitter : public edm::EDProducer 
{

public:
  TrackClusterSplitter(const edm::ParameterSet& iConfig) ;
  ~TrackClusterSplitter() ;
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
  
private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;

  bool simSplitPixel_;
  bool simSplitStrip_;
  bool tmpSplitPixel_;
  bool tmpSplitStrip_;

  // Template splitting needs to know the track direction.
  // We can use either straight tracks, pixel tracks of fully reconstructed track. 
  // Straight tracks go from the first pixel verterx  in the pixel vertex collection 
  // to the cluster center of charge (we use pixel vertices because track vertices are 
  // are not available at this point) 
  // If we set useStraightTracks_ = True, then straight tracks are used
  bool useStraightTracks_;
  
  // If straight track approximation is not used (useStraightTracks_ = False), then, one can use either fully 
  // reconstructed tracks (useTrajectories_ = True) or pixel tracks ((useTrajectories_ = False). 
  // The use of either straight or fully reconstructed tracks give very similar performance. Use straight tracks 
  // by default because it's faster and less involved. Pixel tracks DO NOT work.
  bool useTrajectories_; 

  // These are either "generalTracks", if useTrajectories_ = True, or "pixelTracks" if  useTrajectories_ = False
  edm::EDGetTokenT<TrajTrackAssociationCollection > trajTrackAssociations_;
  edm::EDGetTokenT<std::vector<reco::Track> > tracks_;

  // This is the pixel primary vertex collection
  edm::EDGetTokenT<std::vector<reco::Vertex> > vertices_;
  
  edm::EDGetTokenT< edm::DetSetVector<PixelDigiSimLink> > pixeldigisimlinkToken;
  edm::EDGetTokenT< edm::DetSetVector<StripDigiSimLink> > stripdigisimlinkToken;

  // gavril : what is this for ?
  std::string propagatorName_;
  edm::ESHandle<MagneticField>          magfield_;
  edm::ESHandle<Propagator>             propagator_;
  edm::ESHandle<GlobalTrackingGeometry> geometry_;
  
  // This is needed if we want to to sim/truth pixel splitting
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > pixeldigisimlink;
  
   // This is needed if we want to to sim/truth strip splitting
  edm::Handle< edm::DetSetVector<StripDigiSimLink> > stripdigisimlink;
  

  // Template declarations
  // Pixel templates
  std::vector< SiPixelTemplateStore > thePixelTemp_;
  std::vector< SiPixelTemplateStore2D > thePixelTemp2D_;
  // Strip template
  std::vector< SiStripTemplateStore > theStripTemp_;

  // A pointer to a track and a state on the detector
  struct TrackAndState 
  {
    TrackAndState(const reco::Track *aTrack, TrajectoryStateOnSurface aState) :  
      track(aTrack), state(aState) {}
    const reco::Track*      track;
    TrajectoryStateOnSurface state; 
  };
  
  // A pointer to a cluster and a list of tracks on it
  template<typename Cluster>
  struct ClusterWithTracks 
  {
    ClusterWithTracks(const Cluster &c) : cluster(&c) {}
    const Cluster* cluster;
    std::vector<TrackAndState> tracks;
  };

  typedef ClusterWithTracks<SiPixelCluster> SiPixelClusterWithTracks;
  typedef ClusterWithTracks<SiStripCluster> SiStripClusterWithTracks;


  // a subset of a vector, but with vector-like interface.
  typedef boost::sub_range<std::vector<SiPixelClusterWithTracks> > SiPixelClustersWithTracks;
  typedef boost::sub_range<std::vector<SiStripClusterWithTracks> > SiStripClustersWithTracks;


  // sim strip split
  typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;
  TrackerHitAssociator* hitAssociator;
  
  template<typename C> 
  static const C* getCluster(const TrackingRecHit* hit) ;
  
  template<typename C> 
  static const C* equalClusters(const C &c1, const C &c2) 
  { 
    return nullptr; 
  }
  
  // Find a rechit in a vector of ClusterWithTrack
  template<typename Cluster> class FindCluster 
  {
    
  public:

    FindCluster(const TrackingRecHit* hit) : toFind_( getCluster<Cluster>(hit) ) { } 
    
    bool operator()(const ClusterWithTracks<Cluster> &test) const 
    { 
      assert(test.cluster); // make sure this is not 0
      return test.cluster == toFind_ || equalClusters<Cluster>(*test.cluster, *toFind_); 
    }

  private:

    const Cluster* toFind_;

  };
  
  // Attach tracks to cluster ?!?!
  template<typename Cluster>
  void markClusters(std::map<uint32_t, boost::sub_range<std::vector<ClusterWithTracks<Cluster> > > >& map, 
		    const TrackingRecHit* hit, 
		    const reco::Track* track, 
		    const TrajectoryStateOnSurface& tsos) const ;
  
  template<typename Cluster>
  std::auto_ptr<edmNew::DetSetVector<Cluster> > 
  splitClusters(const std::map<uint32_t, 
		boost::sub_range<std::vector<ClusterWithTracks<Cluster> > > > &input, 
		const reco::Vertex &vtx) const ;
  
  template<typename Cluster>
  void splitCluster(const ClusterWithTracks<Cluster> &cluster, 
		    const GlobalVector &dir, 
		    typename edmNew::DetSetVector<Cluster>::FastFiller &output,
		    DetId detId) const ;
  
  /// working data 
  std::vector<SiPixelClusterWithTracks> allSiPixelClusters;
  std::map<uint32_t, SiPixelClustersWithTracks> siPixelDetsWithClusters; 

  std::vector<SiStripClusterWithTracks> allSiStripClusters;
  std::map<uint32_t, SiStripClustersWithTracks> siStripDetsWithClusters;


};

template<> const SiPixelCluster * TrackClusterSplitter::getCluster<SiPixelCluster>(const TrackingRecHit *hit) ;

template<>
void TrackClusterSplitter::splitCluster<SiPixelCluster> (const SiPixelClusterWithTracks &cluster, 
							 const GlobalVector &dir, 
							 edmNew::DetSetVector<SiPixelCluster>::FastFiller &output,
							 DetId detId
							 ) const ;

template<> const SiStripCluster * TrackClusterSplitter::getCluster<SiStripCluster>(const TrackingRecHit *hit) ;

template<>
void TrackClusterSplitter::splitCluster<SiStripCluster> (const SiStripClusterWithTracks &cluster, 
							 const GlobalVector &dir, 
							 edmNew::DetSetVector<SiStripCluster>::FastFiller &output,
							 DetId detId
							 ) const ;


#define foreach BOOST_FOREACH

TrackClusterSplitter::TrackClusterSplitter(const edm::ParameterSet& iConfig):
  useTrajectories_(iConfig.getParameter<bool>("useTrajectories"))
{
  if (useTrajectories_) {
    trajTrackAssociations_ = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajTrackAssociations"));
  } else {
    propagatorName_ = iConfig.getParameter<std::string>("propagator");
    tracks_ = consumes<std::vector<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"));
  }

  pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"));
  stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("stripClusters"));
  vertices_ = consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("vertices"));

  produces< edmNew::DetSetVector<SiPixelCluster> >();

  produces< edmNew::DetSetVector<SiStripCluster> >();
  
  simSplitPixel_ = (iConfig.getParameter<bool>("simSplitPixel"));
  simSplitStrip_ = (iConfig.getParameter<bool>("simSplitStrip"));
  tmpSplitPixel_ = (iConfig.getParameter<bool>("tmpSplitPixel")); // not so nice... you don't want two bool but some switch
  tmpSplitStrip_ = (iConfig.getParameter<bool>("tmpSplitStrip"));

  useStraightTracks_ = (iConfig.getParameter<bool>("useStraightTracks"));


  if ( simSplitPixel_ ) pixeldigisimlinkToken = consumes< edm::DetSetVector<PixelDigiSimLink> >(edm::InputTag("simSiPixelDigis"));
  if ( simSplitStrip_ ) stripdigisimlinkToken = consumes< edm::DetSetVector<StripDigiSimLink> >(edm::InputTag("simSiStripDigis"));



  /*
    cout << "TrackClusterSplitter : " << endl;
    cout << endl << endl << endl;
    cout << "(int)simSplitPixel_   = " << (int)simSplitPixel_  << endl;
    cout << "(int)simSplitStrip_   = " << (int)simSplitStrip_  << endl;
    cout << "(int)tmpSplitPixel_   = " << (int)tmpSplitPixel_  << endl;
    cout << "(int)tmpSplitStrip_   = " << (int)tmpSplitStrip_  << endl;
    cout << "stripClusters_        = " << stripClusters_        << endl;
    cout << "pixelClusters_        = " << pixelClusters_        << endl;
    cout << "(int)useTrajectories_ = " << (int)useTrajectories_ << endl;
    cout << "trajectories_         = " << trajectories_         << endl;
    cout << "propagatorName_       = " << propagatorName_       << endl;
    cout << "vertices_             = " << vertices_             << endl;
    cout << "useStraightTracks_    = " << useStraightTracks_    << endl;
    cout << endl << endl << endl;
  */

  // Load template; 40 for barrel and 41 for endcaps
  SiPixelTemplate::pushfile( 40, thePixelTemp_ );
  SiPixelTemplate::pushfile( 41, thePixelTemp_ );
  SiPixelTemplate2D::pushfile( 40, thePixelTemp2D_ );
  SiPixelTemplate2D::pushfile( 41, thePixelTemp2D_ );

  // Load strip templates
  SiStripTemplate::pushfile( 11, theStripTemp_ );
  SiStripTemplate::pushfile( 12, theStripTemp_ );
  SiStripTemplate::pushfile( 13, theStripTemp_ );
  SiStripTemplate::pushfile( 14, theStripTemp_ );
  SiStripTemplate::pushfile( 15, theStripTemp_ );
  SiStripTemplate::pushfile( 16, theStripTemp_ );

}


template<>
const SiStripCluster*
TrackClusterSplitter::getCluster<SiStripCluster>(const TrackingRecHit* hit) 
{
  if ( typeid(*hit) == typeid(SiStripRecHit2D) ) 
    {
      return (static_cast<const SiStripRecHit2D &>(*hit)).cluster().get();
    } 
  else if ( typeid(*hit) == typeid(SiStripRecHit1D) ) 
    {
      return (static_cast<const SiStripRecHit1D &>(*hit)).cluster().get();
    } 
  else 
    throw cms::Exception("Unsupported") << "Detid of type " << typeid(*hit).name() << " not supported.\n";
}

template<>
const SiPixelCluster*
TrackClusterSplitter::getCluster<SiPixelCluster>(const TrackingRecHit* hit) 
{
  if ( typeid(*hit) == typeid(SiPixelRecHit) ) 
    {
      return (static_cast<const SiPixelRecHit&>(*hit)).cluster().get();
    } 
  else 
    throw cms::Exception("Unsupported") << "Detid of type " << typeid(*hit).name() << " not supported.\n";
}

TrackClusterSplitter::~TrackClusterSplitter()
{
}

void
TrackClusterSplitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);  
  
  if ( !useTrajectories_ ) 
    {
      iSetup.get<IdealMagneticFieldRecord>().get( magfield_ );
      iSetup.get<TrackingComponentsRecord>().get( "AnalyticalPropagator", propagator_ );
    }
 
  Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
  Handle<edmNew::DetSetVector<SiStripCluster> > inputStripClusters;
  
  iEvent.getByToken(pixelClusters_, inputPixelClusters);
  
  iEvent.getByToken(stripClusters_, inputStripClusters);
  
  if(simSplitStrip_)
    hitAssociator = new TrackerHitAssociator(iEvent);

    
  allSiPixelClusters.clear(); siPixelDetsWithClusters.clear();
  allSiStripClusters.clear(); siStripDetsWithClusters.clear();

  
  allSiPixelClusters.reserve(inputPixelClusters->dataSize()); // this is important, otherwise push_back invalidates the iterators
  allSiStripClusters.reserve(inputStripClusters->dataSize()); // this is important, otherwise push_back invalidates the iterators

  
  // fill in the list of all tracks
  foreach(const edmNew::DetSet<SiPixelCluster> &ds, *inputPixelClusters) 
    {
      std::vector<SiPixelClusterWithTracks>::iterator start = allSiPixelClusters.end();
      allSiPixelClusters.insert(start, ds.begin(), ds.end());
    
      std::vector<SiPixelClusterWithTracks>::iterator end   = allSiPixelClusters.end();
      siPixelDetsWithClusters[ds.detId()] = SiPixelClustersWithTracks(start,end);
    }
 
  foreach(const edmNew::DetSet<SiStripCluster> &ds, *inputStripClusters) 
    {
      std::vector<SiStripClusterWithTracks>::iterator start = allSiStripClusters.end();
      allSiStripClusters.insert(start, ds.begin(), ds.end());
     
      std::vector<SiStripClusterWithTracks>::iterator end   = allSiStripClusters.end();
      siStripDetsWithClusters[ds.detId()] = SiStripClustersWithTracks(start,end);
    }
  
  if ( useTrajectories_ ) 
    { 
      // Here use the fully reconstructed tracks to get the track angle

      Handle<TrajTrackAssociationCollection> trajectories; 
      iEvent.getByToken(trajTrackAssociations_, trajectories);
      for ( TrajTrackAssociationCollection::const_iterator it = trajectories->begin(), 
	      ed = trajectories->end(); it != ed; ++it ) 
	{ 
	  const Trajectory  & traj =  *it->key;
	  const reco::Track * tk   = &*it->val;
	  
	  if ( traj.measurements().size() != tk->recHitsSize() ) 
	    throw cms::Exception("Aargh") << "Sizes don't match: traj " << traj.measurements().size()  
					  << ", tk " << tk->recHitsSize() << "\n";
	  
	  trackingRecHit_iterator it_hit = tk->recHitsBegin(), ed_hit = tk->recHitsEnd();
	  
	  const Trajectory::DataContainer & tms = traj.measurements();
	  
	  size_t i_hit = 0, last_hit = tms.size()-1;
	  
	  bool first = true, reversed = false;
	  
	  for (; it_hit != ed_hit; ++it_hit, ++i_hit) 
	    {
	      // ignore hits with no detid

	      if ((*it_hit)->geographicalId().rawId() == 0)
		{
		  //cout << "It should never happen that a trackingRecHit has no detector ID !!!!!!!!!!!!!!!!! " << endl;
		  continue;
		}

	      // if it's the first hit, check the ordering of track vs trajectory
	      if (first) 
		{
		  
		  if ((*it_hit)->geographicalId() == tms[i_hit].recHit()->hit()->geographicalId()) 
		    {
		      reversed = false;
                    } 
		  else if ((*it_hit)->geographicalId() == tms[last_hit-i_hit].recHit()->hit()->geographicalId()) 
		    {
		      reversed = true;
                    } 
		  else 
		    {
		      throw cms::Exception("Aargh") << "DetIDs don't match either way :-( \n";
                    }
                }  

	      const TrackingRecHit *hit = it_hit->get();
	      if ( hit == 0 || !hit->isValid() )
		continue;
	      
	      int subdet = hit->geographicalId().subdetId();
	      
	      if (subdet >= 3) 
		{ // strip
		  markClusters<SiStripCluster>(siStripDetsWithClusters, hit, tk, tms[reversed ? last_hit-i_hit : i_hit].updatedState());
                } 
	      else if (subdet >= 1) 
		{ // pixel
		  markClusters<SiPixelCluster>(siPixelDetsWithClusters, hit, tk, tms[reversed ? last_hit-i_hit : i_hit].updatedState());
                } 
	      else 
		{
		  edm::LogWarning("HitNotFound") << "Hit of type " << typeid(*hit).name() << ",  detid " 
						 << hit->geographicalId().rawId() << ", subdet " << subdet;
                }
            }
        }
    } 
  else 
    {
      // Here use the pixel tracks to get the track angles

      Handle<std::vector<reco::Track> > tracks; 
      iEvent.getByToken(tracks_, tracks);
      //TrajectoryStateTransform transform;
      foreach (const reco::Track &track, *tracks) 
	{
	  FreeTrajectoryState atVtx   =  trajectoryStateTransform::innerFreeState(track, &*magfield_);
	  trackingRecHit_iterator it_hit = track.recHitsBegin(), ed_hit = track.recHitsEnd();
	  for (; it_hit != ed_hit; ++it_hit) 
	    {
	      const TrackingRecHit *hit = it_hit->get();
	      if ( hit == 0 || !hit->isValid() ) 
		continue;
	      
	      int subdet = hit->geographicalId().subdetId();

	      if ( subdet == 0 ) 
		continue;
	      
	      const GeomDet *det = geometry_->idToDet( hit->geographicalId() );
	      
	      if ( det == 0 ) 
		{
		  edm::LogError("MissingDetId") << "DetIDs " << (int)(hit->geographicalId()) << " is not in geometry.\n";
		  continue;
                }
	      
	      TrajectoryStateOnSurface prop = propagator_->propagate(atVtx, det->surface());
	      if ( subdet >= 3 ) 
		{ // strip
		  markClusters<SiStripCluster>(siStripDetsWithClusters, hit, &track, prop);
                } 
	      else if (subdet >= 1) 
		{ // pixel
		  markClusters<SiPixelCluster>(siPixelDetsWithClusters, hit, &track, prop);
                } 
	      else 
		{
		  edm::LogWarning("HitNotFound") << "Hit of type " << typeid(*hit).name() << ",  detid " 
						 << hit->geographicalId().rawId() << ", subdet " << subdet;
		}
            }
        }
    }

  Handle<std::vector<reco::Vertex> > vertices; 
  iEvent.getByToken(vertices_, vertices);
  
  // Needed in case of simsplit
  if ( simSplitPixel_ )
    iEvent.getByToken(pixeldigisimlinkToken, pixeldigisimlink);
    
  // Needed in case of strip simsplit
  if ( simSplitStrip_ )
    iEvent.getByToken(stripdigisimlinkToken, stripdigisimlink);

  // gavril : to do: choose the best vertex here instead of just choosing the first one ? 
  std::auto_ptr<edmNew::DetSetVector<SiPixelCluster> > newPixelClusters( splitClusters( siPixelDetsWithClusters, vertices->front() ) );
  std::auto_ptr<edmNew::DetSetVector<SiStripCluster> > newStripClusters( splitClusters( siStripDetsWithClusters, vertices->front() ) );
  
  if ( simSplitStrip_ )
    delete hitAssociator;

  iEvent.put(newPixelClusters);
  iEvent.put(newStripClusters);
    
  allSiPixelClusters.clear(); siPixelDetsWithClusters.clear();
  allSiStripClusters.clear(); siStripDetsWithClusters.clear();

}

template<typename Cluster>
void TrackClusterSplitter::markClusters( std::map<uint32_t, boost::sub_range<std::vector<ClusterWithTracks<Cluster> > > >& map, 
					 const TrackingRecHit* hit, 
					 const reco::Track* track, 
					 const TrajectoryStateOnSurface& tsos) const 
{
  boost::sub_range<std::vector<ClusterWithTracks<Cluster> > >& range = map[hit->geographicalId().rawId()];
  
  typedef typename std::vector<ClusterWithTracks<Cluster> >::iterator IT;
  IT match = std::find_if(range.begin(), range.end(), FindCluster<Cluster>(hit));
  
  if ( match != range.end() ) 
    {
      match->tracks.push_back( TrackAndState(track,tsos) );
    } 
  else 
    {
      edm::LogWarning("ClusterNotFound") << "Cluster of type " << typeid(Cluster).name() << " on detid " 
					 << hit->geographicalId().rawId() << " from hit of type " << typeid(*hit).name();
    }
}

template<typename Cluster>
std::auto_ptr<edmNew::DetSetVector<Cluster> > 
TrackClusterSplitter::splitClusters(const std::map<uint32_t, boost::sub_range<std::vector<ClusterWithTracks<Cluster> > > > &input, 
				    const reco::Vertex &vtx) const 
{
  std::auto_ptr<edmNew::DetSetVector<Cluster> > output(new edmNew::DetSetVector<Cluster>());
  typedef std::pair<uint32_t, boost::sub_range<std::vector<ClusterWithTracks<Cluster> > > > pair;
  
  foreach(const pair &p, input) 
    {
      const GeomDet* det = geometry_->idToDet( DetId(p.first) );
      
      if ( det == 0 ) 
      	{ 
      	  edm::LogError("MissingDetId") << "DetIDs " << p.first << " is not in geometry.\n";
	  continue;
	}
      
      // gavril: Pass the PV instead of direction
      // GlobalVector dir(det->position().x() - vtx.x(), det->position().y() - vtx.y(), det->position().z() - vtx.z());
      GlobalVector primary_vtx( vtx.x(), vtx.y(), vtx.z() );

      // Create output collection
      typename edmNew::DetSetVector<Cluster>::FastFiller detset(*output, p.first);
      
      // fill it
      foreach(const ClusterWithTracks<Cluster> &c, p.second) 
	{
	  splitCluster(c, primary_vtx, detset, DetId(p.first) );
        }
    }

  return output;
}

template<typename Cluster>
void TrackClusterSplitter::splitCluster(const ClusterWithTracks<Cluster> &cluster, 
					const GlobalVector &dir, 
					typename edmNew::DetSetVector<Cluster>::FastFiller &output, 
					DetId detId) const 
{
  //cout << "Should never be here: TrackClusterSplitter, TrackClusterSplitter::splitCluster(...)  !!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
  throw cms::Exception("LogicError", "This should not be called");
}

template<>
void TrackClusterSplitter::splitCluster<SiStripCluster> (const SiStripClusterWithTracks& c, 
							 const GlobalVector &vtx, 
							 edmNew::DetSetVector<SiStripCluster>::FastFiller &output,
							 DetId detId
							 ) const 
{ 
  if ( simSplitStrip_ )
    {      
      bool cluster_was_successfully_split = false;
            
      const SiStripCluster* clust = static_cast<const SiStripCluster*>(c.cluster);
      
      std::vector<SimHitIdpr> associatedIdpr;
      
      hitAssociator->associateSimpleRecHitCluster(clust, detId, associatedIdpr);

      size_t splittableClusterSize = 0;
      splittableClusterSize = associatedIdpr.size();
      std::vector<uint8_t> amp = clust->amplitudes();
      int clusiz = amp.size();
      associatedIdpr.clear();

      SiStripDetId ssdid( detId );

      // gavril : sim splitting can be applied to the forward detectors as well...

      if ( ( splittableClusterSize > 1 && amp.size() > 2 ) && 
	   ( (int)ssdid.moduleGeometry() == 1 || 
	     (int)ssdid.moduleGeometry() == 2 || 
	     (int)ssdid.moduleGeometry() == 3 || 
	     (int)ssdid.moduleGeometry() == 4 ) )
	{
	  
	  edm::DetSetVector<StripDigiSimLink>::const_iterator isearch = stripdigisimlink->find(detId);
	  
	  int first  = clust->firstStrip();
	  int last   = first + clusiz;
	  uint16_t rawAmpl = 0, currentAmpl = 0;
	  
	  std::vector<uint16_t> tmp1, tmp2;
	  
	  std::vector<int> firstStrip;
	  std::vector<bool> trackInStrip;
	  std::vector<unsigned int> trackID;
	  std::vector<float> trackFraction;
	  std::vector< std::vector<uint16_t> > trackAmp;
	  unsigned int currentChannel( 9999 );
	  unsigned int thisTrackID = 0;

	  if ( isearch != stripdigisimlink->end() ) 
	    {
	      edm::DetSet<StripDigiSimLink> link_detset = (*isearch);
	      
	      for ( edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin();
		  linkiter != link_detset.data.end(); linkiter++)
		{
		  if ( (int)(linkiter->channel()) >= first  && (int)(linkiter->channel()) < last  )
		    {
		      int stripIdx = (int)linkiter->channel()-first;
		      rawAmpl = (uint16_t)(amp[stripIdx]);  
		      
		      // DigiSimLinks are ordered first by channel; there can be > 1 track, and > 1 simHit for each track
		      
		      if ( linkiter->channel() != currentChannel ) 
			{
			  // New strip; store amplitudes for the previous one
			  uint16_t thisAmpl;
			  
			  for (size_t i=0; i<trackID.size(); ++i)  
			    {
			      if ( trackInStrip[i] ) 
				{
				  if ( ( thisAmpl=currentAmpl ) < 254 )
				    thisAmpl = min( uint16_t(253), max(uint16_t(0), (uint16_t)(currentAmpl*trackFraction[i]+0.5)) );
				  trackAmp[i].push_back( thisAmpl );
				}
			      
			      trackFraction[i] = 0;
			      trackInStrip[i] = false;
			    }
			  
			  currentChannel = linkiter->channel();
			  currentAmpl = rawAmpl;
			}
		      
		      // Now deal with this new DigiSimLink
		      thisTrackID = linkiter->SimTrackId();
		      
		      // Have we seen this track yet?
		      bool newTrack = true;
		      int thisTrackIdx = 9999;
		      
		      for (size_t i=0; i<trackID.size(); ++i) 
			{
			  if ( trackID[i] == thisTrackID ) 
			    {
			      thisTrackIdx = i;
			      newTrack = false;
			    }
			}
		     
		      if ( newTrack ) 
			{
			  trackInStrip.push_back(false);  // We'll set it true below
			  trackID.push_back(thisTrackID);
			  firstStrip.push_back(currentChannel);
			  std::vector<uint16_t> ampTmp;
			  trackAmp.push_back(ampTmp);
			  trackFraction.push_back(0);
			  thisTrackIdx = trackID.size()-1;
			}
		      
		      trackInStrip[thisTrackIdx] = true;
		      trackFraction[thisTrackIdx] += linkiter->fraction();
		      currentAmpl = rawAmpl;
		       
		    }
		  
		}// end of loop over DigiSimLinks
		  
	      // we want to continue here!!!!
	      
	      std::vector<SiStripCluster> newCluster;
	      // Fill amplitudes for the last strip and create a cluster for each track
	      uint16_t thisAmpl;
	      
	      for (size_t i=0; i < trackID.size(); ++i) 
		{ 
		  if ( trackInStrip[i] ) 
		    {
		      if ( ( thisAmpl=rawAmpl ) < 254 ) 
			thisAmpl = min(uint16_t(253), max(uint16_t(0), (uint16_t)(rawAmpl*trackFraction[i]+0.5)));
		      
		      if ( thisAmpl > 0 )
			trackAmp[i].push_back( thisAmpl );
		      //else
		      //cout << "thisAmpl = " << (int)thisAmpl << endl;
		    }
		  
		  newCluster.push_back( SiStripCluster( 
							firstStrip[i],
							trackAmp[i].begin(), 
							trackAmp[i].end() ) );
		  
		}
	      
	      
	      for ( size_t i=0; i<newCluster.size(); ++i )
		{
		  
		  // gavril : This is never used. Will use it below 
		  float clusterAmp = 0.0;
		  for (size_t j=0; j<trackAmp[i].size(); ++j ) 
		    clusterAmp += (float)(trackAmp[i])[j];
		  
		  if ( clusterAmp > 0.0 && firstStrip[i] != 9999 && trackAmp[i].size() > 0  ) 
		    { 
		      // gavril :  I think this should work
		      output.push_back( newCluster[i] );
		      
		      //cout << endl << endl << endl; 
		      //cout << "(int)(newCluster[i].amplitudes().size()) = " << (int)(newCluster[i].amplitudes().size()) << endl;
		      //for ( int j=0; j<(int)(newCluster[i].amplitudes().size()); ++j )
		      //cout << "(newCluster[i].amplitudes())[j] = " << (int)(newCluster[i].amplitudes())[j] << endl;
		      
		      cluster_was_successfully_split = true;
		    } 
		  else 
		    {
		      //std::cout << "\t\t Rejecting new cluster" << std::endl;
		      
		      // gavril : I think this pointer should be deleted above
		      //delete newCluster[i];
		    }
		}
	            
	    } // if ( isearch != stripdigisimlink->end() ) ... 
	  else 
	    {
	      // Do nothing...
	    }
	}
      
      
      if ( !cluster_was_successfully_split )
	output.push_back( *c.cluster ); 
      
    } // end of   if ( strip_simSplit_ )...
  
  else if ( tmpSplitStrip_ )
   {
      bool cluster_was_successfully_split = false;
      
      const SiStripCluster* theStripCluster = static_cast<const SiStripCluster*>(c.cluster);
      
      if ( theStripCluster )
	{
	  //SiStripDetId ssdid( theStripCluster->geographicalId() );
	  SiStripDetId ssdid( detId.rawId() );

	  // Do not attempt to split clusters of size less than or equal to one.
	  // Only split clusters in IB1, IB2, OB1, OB2 (TIB and TOB).

	  if ( (int)theStripCluster->amplitudes().size() <= 1 || 
	       ( (int)ssdid.moduleGeometry() != 1 && 
		 (int)ssdid.moduleGeometry() != 2 && 
		 (int)ssdid.moduleGeometry() != 3 && 
		 (int)ssdid.moduleGeometry() != 4 ) )
	    {    
	      // Do nothing.
	      //cout << endl;
	      //cout << "Will NOT attempt to split this clusters: " << endl;
	      //cout << "(int)theStripCluster->amplitudes().size() = " << (int)theStripCluster->amplitudes().size() << endl;
	      //cout << "(int)ssdid.moduleGeometry() = " << (int)ssdid.moduleGeometry() << endl;
	      
	    }
	  else // if ( (int)theStripCluster->amplitudes().size() <= 1 )
	    {
	     
	      //cout << endl;
	      //cout << "Will attempt to split this clusters: " << endl;

	      int ID = -99999;
	     	     
	      int is_stereo = (int)( ssdid.stereo() ); 
	      
	      if      ( ssdid.moduleGeometry() == 1 ) // IB1 
		{
		  if ( !is_stereo ) 
		    ID = 11;
		  else
		    ID = 12;
		}
	      else if ( ssdid.moduleGeometry() == 2 ) // IB2
		{
		  ID = 13;
		}
	      else if ( ssdid.moduleGeometry() == 3 ) // OB1
		{
		  ID = 16; 
		}
	      else if ( ssdid.moduleGeometry() == 4 ) // OB2
		{
		  if ( !is_stereo )
		    ID = 14;
		  else
		    ID = 15;
		}
	      else 
		{
		  throw cms::Exception("TrackClusterSplitter::splitCluster") 
		    << "\nERROR: Wrong strip teplate ID. Should only use templates for IB1, IB2, OB1 and OB2 !!!" << "\n\n";
		}
	      



	      // Begin: determine incident angles ============================================================
	      
	      float cotalpha_ = -99999.9;
	      float cotbeta_  = -99999.9;
	      
	      // First, determine track angles from straight track approximation
	      
	      // Find crude cluster center
	      // gavril : This is in local coordinates ? 
	      float xcenter = theStripCluster->barycenter();
	     
	      const GeomDetUnit* theDet = geometry_->idToDetUnit( detId );
	      const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>( theDet );
		      
	      if ( !stripDet )
		{
		  throw cms::Exception("TrackClusterSplitter : ") 
		    << "\nERROR: Wrong stripDet !!! " << "\n\n";
		}


	      const StripTopology* theTopol = &( stripDet->specificTopology() );

	      // Transform from measurement to local coordinates (in cm)
	      // gavril: may have to differently if kicks/bows are introduced. However, at this point there are no tracks...:
	      // LocalPoint lp = theTopol->localPosition( xcenter, /*const Topology::LocalTrackPred & */ trkPred );

	      // gavril : What is lp.y() for strips ? It is zero, but is that the strip center or one of the ends ? 
	      LocalPoint lp = theTopol->localPosition( xcenter );
	      
	      // Transform from local to global coordinates
	      GlobalPoint gp0 = theDet->surface().toGlobal( lp );
	      
	       // Make a vector pointing from the PV to the cluster center 
	      GlobalPoint gp(gp0.x()-vtx.x(), gp0.y()-vtx.y(), gp0.z()-vtx.z() );
	      
	      // Make gp a unit vector, gv, pointing from the PV to the cluster center
	      float gp_mod = sqrt( gp.x()*gp.x() + gp.y()*gp.y() + gp.z()*gp.z() );
	      float gpx = gp.x()/gp_mod;
	      float gpy = gp.y()/gp_mod;
	      float gpz = gp.z()/gp_mod;
	      GlobalVector gv(gpx, gpy, gpz);

	      // Make unit vectors in local coordinates and then transform them in global coordinates
	      const Local3DVector lvx(1.0, 0.0, 0.0);
	      GlobalVector gvx = theDet->surface().toGlobal( lvx );
	      const Local3DVector lvy(0.0, 1.0, 0.0);
	      GlobalVector gvy = theDet->surface().toGlobal( lvy );
	      const Local3DVector lvz(0.0, 0.0, 1.0);
	      GlobalVector gvz = theDet->surface().toGlobal( lvz );
	      
	      // Calculate angles alpha and beta
	      float gv_dot_gvx = gv.x()*gvx.x() + gv.y()*gvx.y() + gv.z()*gvx.z();
	      float gv_dot_gvy = gv.x()*gvy.x() + gv.y()*gvy.y() + gv.z()*gvy.z();
	      float gv_dot_gvz = gv.x()*gvz.x() + gv.y()*gvz.y() + gv.z()*gvz.z();
	      
	      // gavril : check beta !!!!!!!!!!!!
	      //float alpha_ = atan2( gv_dot_gvz, gv_dot_gvx );
	      //float beta_  = atan2( gv_dot_gvz, gv_dot_gvy );
	      
	      cotalpha_ = gv_dot_gvx / gv_dot_gvz;
	      cotbeta_  = gv_dot_gvy / gv_dot_gvz;
	       
	      // Attempt to get a better angle from tracks (either pixel tracks or full tracks)
	      if ( !useStraightTracks_ )
		{
		  //cout << "TrackClusterSplitter.cc : " << endl;
		  //cout << "Should not be here for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;

		  // Use either pixel tracks (useTrajectories_ = False) or fully reconstructed tracks (useTrajectories_ = True)
		  // When pixel/full tracks are not associated with the current cluster, will use angles from straight tracks
		  
		  // These are the tracks associated with this cluster
		  std::vector<TrackAndState> vec_tracks_states = c.tracks;

		  if ( (int)vec_tracks_states.size() > 0 )
		    {
		      //cout << "There is at least one track associated with this cluster. Pick the one with largest Pt." << endl;
		      //cout << "(int)vec_tracks_states.size() = " << (int)vec_tracks_states.size() << endl;
		      
		      int index_max_pt = -99999;  // index of the track with the highest Pt
		      float max_pt = -99999.9;
		      
		      for (int i=0; i<(int)vec_tracks_states.size(); ++i )
			{
			  const reco::Track* one_track = vec_tracks_states[i].track;
			  
			  if ( one_track->pt() > max_pt )
			    {
			      index_max_pt = i;
			      max_pt = one_track->pt();
			    }  
			}
		      
		      // Pick the tsos from the track with highest Pt
		      // gavril: Should we use highest Pt or best Chi2 ?
		      TrajectoryStateOnSurface one_tsos = vec_tracks_states[index_max_pt].state;
		      
		      LocalTrajectoryParameters ltp = one_tsos.localParameters();
		      
		      LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
		      
		      float locx = localDir.x();
		      float locy = localDir.y();
		      float locz = localDir.z();
		      
		      //alpha_ = atan2( locz, locx );
		      //beta_  = atan2( locz, locy );
		      
		      cotalpha_ = locx/locz;
		      cotbeta_  = locy/locz;
		      
		    } // if ( (int)vec_tracks_states.size() > 0 )
		      
		} // if ( !useStraightTracks_ )


	      // End: determine incident angles ============================================================


	      // Calculate strip cluster charge and store amplitudes in vector for later use
	      
	      //cout << endl;
	      //cout << "Calculate strip cluster charge and store amplitudes in vector for later use" << endl;

	      float strip_cluster_charge = 0.0;
	      std::vector<float> vec_cluster_charge; 
	      vec_cluster_charge.clear();
	      int cluster_size = (int)( (theStripCluster->amplitudes()).size() );
	      
	      int cluster_charge = 0;
	      for (int i=0; i<cluster_size; ++i)
		{
		  float current_strip_charge = (float)( (theStripCluster->amplitudes())[i] );

		  strip_cluster_charge += current_strip_charge;
		  vec_cluster_charge.push_back( current_strip_charge );  
		
		  cluster_charge +=current_strip_charge; 
		}
	     
	   
	      //cout << endl;
	      //cout << "Calling strip qbin to see if the strip cluster has to be split..." << endl;	      
              SiStripTemplate strip_templ_(theStripTemp_);
	      int strip_templQbin_ = strip_templ_.qbin( ID, cotalpha_, cotbeta_, strip_cluster_charge );

	      if ( strip_templQbin_ < 0 || strip_templQbin_ > 5 )
		{
		  // Do nothing...
		  // cout << "Wrong strip strip_templQbin_ = " << strip_templQbin_ << endl;
		} // if ( strip_templQbin_ < 0 || strip_templQbin_ > 5 )
	      else // if ( strip_templQbin_ < 0 || strip_templQbin_ > 5 ) {...} else
		{
		  if ( strip_templQbin_ != 0 ) 
		    {
		      // Do not split this cluster. Do nothing.
		      //cout << endl;
		      //cout << "Do NOT split this cluster" << endl;

		    } // if ( strip_templQbin_ != 0 ) 
		  else // if ( templQbin_ != 0 ) {...} else
		    {
		      //cout << endl;
		      //cout << "Split this cluster" << endl;

		      // gavril : Is this OK ?
		      uint16_t first_strip = theStripCluster->firstStrip();
 
		      LocalVector lbfield = ( stripDet->surface() ).toLocal( magfield_->inTesla( stripDet->surface().position() ) ); 
		      float locBy = lbfield.y();
		      
		      // Initialize values coming back from SiStripTemplateSplit::StripTempSplit
		      float stripTemplXrec1_  = -99999.9;
		      float stripTemplXrec2_  = -99999.9;
		      float stripTemplSigmaX_ = -99999.9;
		      float stripTemplProbX_  = -99999.9;
		      int   stripTemplQbin_   = -99999;
		      float stripTemplProbQ_  = -99999.9;


		      /*
			cout << endl;
			cout << "About to call SiStripTemplateSplit::StripTempSplit(...)" << endl;
			
			cout << endl;
			cout << "ID        = " << ID        << endl;
			cout << "cotalpha_ = " << cotalpha_ << endl;
			cout << "cotbeta_  = " << cotbeta_  << endl;
			cout << "locBy     = " << locBy     << endl;
			cout << "Amplitudes: "; 
			for (int i=0; i<(int)vec_cluster_charge.size(); ++i)
			cout << vec_cluster_charge[i] << ", ";
			cout << endl;
		      */

		      int ierr =
			SiStripTemplateSplit::StripTempSplit(ID, 
							     cotalpha_, cotbeta_, 
							     locBy, 
							     vec_cluster_charge, 
							     strip_templ_, 
							     stripTemplXrec1_, 
							     stripTemplXrec2_, 
							     stripTemplSigmaX_, 
							     stripTemplProbX_,
							     stripTemplQbin_,
							     stripTemplProbQ_ );
		      
		     

		      stripTemplXrec1_ += 2*strip_templ_.xsize();
		      stripTemplXrec2_ += 2*strip_templ_.xsize();
		      
		    

		      if ( ierr != 0 )
			{
			  //cout << endl;
			  //cout << "Strip cluster splitting failed: ierr = " << ierr << endl;
			}
		      else // if ( ierr != 0 )
			{
			  // Cluster was successfully split. 
			  // Make the two split clusters and put them in the split cluster collection
			  
			  //cout << endl; 
			  //cout << "Cluster was successfully split" << endl;

			  cluster_was_successfully_split = true;

			  std::vector<float> strip_cluster1;
			  std::vector<float> strip_cluster2;

			  strip_cluster1.clear();
			  strip_cluster2.clear();
			  
			  // gavril : Is this OK ?!?!?!?! 
			  for (int i=0; i<BXSIZE; ++i)
			    {
			      strip_cluster1.push_back(0.0);
			      strip_cluster2.push_back(0.0);
			    }

			  //cout << endl;
			  //cout << "About to call interpolate and sxtemp" << endl;

			  strip_templ_.interpolate(ID, cotalpha_, cotbeta_, locBy);
			  strip_templ_.sxtemp(stripTemplXrec1_, strip_cluster1);
			  strip_templ_.sxtemp(stripTemplXrec2_, strip_cluster2);
			 
			 
			  
			  vector<SiStripDigi> vecSiStripDigi1;
			  vecSiStripDigi1.clear();  
			  int strip_cluster1_size = (int)strip_cluster1.size();
			  for (int i=2; i<strip_cluster1_size; ++i)
			    {
			      if ( strip_cluster1[i] > 0.0 )
				{
				  SiStripDigi current_digi1( first_strip + i-2, strip_cluster1[i] ); 
				  
				  vecSiStripDigi1.push_back( current_digi1 );
				}
			    }
			   
			  
			  
			  vector<SiStripDigi> vecSiStripDigi2;
			  vecSiStripDigi2.clear();  
			  int strip_cluster2_size = (int)strip_cluster2.size();
			  for (int i=2; i<strip_cluster2_size; ++i)
			    {
			      if ( strip_cluster2[i] > 0.0 )
				{
				  SiStripDigi current_digi2( first_strip + i-2, strip_cluster2[i] ); 
				  
				  vecSiStripDigi2.push_back( current_digi2 );
				}
			    }
			  
			  
			
			  
			  std::vector<SiStripDigi>::const_iterator SiStripDigiIterBegin1 = vecSiStripDigi1.begin();
			  std::vector<SiStripDigi>::const_iterator SiStripDigiIterEnd1   = vecSiStripDigi1.end();
			  std::pair<std::vector<SiStripDigi>::const_iterator, 
			    std::vector<SiStripDigi>::const_iterator> SiStripDigiRange1 
			    = make_pair(SiStripDigiIterBegin1, SiStripDigiIterEnd1);

			  //if ( SiStripDigiIterBegin1 == SiStripDigiIterEnd1 )
			  //{
			  //  throw cms::Exception("TrackClusterSplitter : ") 
			  //<< "\nERROR: SiStripDigiIterBegin1 = SiStripDigiIterEnd1 !!!" << "\n\n";
			  //}
			  
			  std::vector<SiStripDigi>::const_iterator SiStripDigiIterBegin2 = vecSiStripDigi2.begin();
			  std::vector<SiStripDigi>::const_iterator SiStripDigiIterEnd2   = vecSiStripDigi2.end();
			  std::pair<std::vector<SiStripDigi>::const_iterator, 
			    std::vector<SiStripDigi>::const_iterator> SiStripDigiRange2 
			    = make_pair(SiStripDigiIterBegin2, SiStripDigiIterEnd2);

			  // Sanity check...
			  //if ( SiStripDigiIterBegin2 == SiStripDigiIterEnd2 )
			  //{
			  //  cout << endl;
			  //  cout << "SiStripDigiIterBegin2 = SiStripDigiIterEnd2 !!!!!!!!!!!!!!!" << endl;
			  //}
			  
			  
			  // Save the split clusters

			  if ( SiStripDigiIterBegin1 != SiStripDigiIterEnd1 )
			    {    
			      // gavril : Raw id ?
			      SiStripCluster cl1( SiStripDigiRange1 );

			      cl1.setSplitClusterError( stripTemplSigmaX_ );

			      output.push_back( cl1 );
			    
			      if ( (int)cl1.amplitudes().size() <= 0 )
				{
				  throw cms::Exception("TrackClusterSplitter : ") 
				    << "\nERROR: (1) Wrong split cluster of size = " << (int)cl1.amplitudes().size() << "\n\n";
				}
			      
			    } // if ( SiStripDigiIterBegin1 != SiStripDigiIterEnd1 )

			  if ( SiStripDigiIterBegin2 != SiStripDigiIterEnd2 )
			    {
			      // gavril : Raw id ?
			      SiStripCluster cl2( SiStripDigiRange2 );
			      cl2.setSplitClusterError( stripTemplSigmaX_ );
			      output.push_back( cl2 ); 
		
			      if ( (int)cl2.amplitudes().size() <= 0 )
				{
				  throw cms::Exception("TrackClusterSplitter : ") 
				    << "\nERROR: (2) Wrong split cluster of size = " << (int)cl2.amplitudes().size() << "\n\n";
				}
			      
			    } // if ( SiStripDigiIterBegin2 != SiStripDigiIterEnd2 )
 
			
			   
			} // else // if ( ierr != 0 )

		    } // else // if ( strip_templQbin_ != 0 ) {...} else 
		  
		} // else // if ( strip_templQbin_ < 0 || strip_templQbin_ > 5 ) {...} else

	    } // else // if ( (int)theStripCluster->amplitudes().size() <= 1 )

	  
	  if ( !cluster_was_successfully_split )
	    output.push_back( *c.cluster );

	} // if ( theStripCluster )
      else
	{
	  throw cms::Exception("TrackClusterSplitter : ") 
	    << "\nERROR: This is not a SiStripCluster  !!!" << "\n\n";
	}

    } // else if ( strip_tmpSplit_ )
  else
    {
      // gavril : Neither sim nor template splitter. Just add the cluster as it was. 
      output.push_back( *c.cluster );
    }
}

template<>
void TrackClusterSplitter::splitCluster<SiPixelCluster> (const SiPixelClusterWithTracks &c, 
							 const GlobalVector &vtx, 
							 edmNew::DetSetVector<SiPixelCluster>::FastFiller &output, 
							 DetId detId 
							 ) const 
{ 
  // The sim splitter:
  if ( simSplitPixel_ ) 
    {
      // cout << "Cluster splitting using simhits " << endl;
 
      int minPixelRow = (*c.cluster).minPixelRow();
      int maxPixelRow = (*c.cluster).maxPixelRow();
      int minPixelCol = (*c.cluster).minPixelCol();
      int maxPixelCol = (*c.cluster).maxPixelCol();    
      int dsl = 0; // number of digisimlinks
      
      edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = pixeldigisimlink->find(output.id());
      if (isearch != pixeldigisimlink->end()){
	edm::DetSet<PixelDigiSimLink> digiLink = (*isearch);
	
	edm::DetSet<PixelDigiSimLink>::const_iterator linkiter = digiLink.data.begin();
	//create a vector for the track ids in the digisimlinks
	std::vector<int> simTrackIdV;  
	simTrackIdV.clear();
	//create a vector for the new splittedClusters 
	std::vector<SiPixelCluster> splittedCluster;
	splittedCluster.clear();
	
	for ( ; linkiter != digiLink.data.end(); linkiter++) 
	  { // loop over all digisimlinks 
	    dsl++;
	    std::pair<int,int> pixel_coord = PixelDigi::channelToPixel(linkiter->channel());
	    
	    // is the digisimlink inside the cluster boundaries?
	    if ( pixel_coord.first  <= maxPixelRow && 
		 pixel_coord.first  >= minPixelRow &&
		 pixel_coord.second <= maxPixelCol &&
		 pixel_coord.second >= minPixelCol ) 
	      {
		bool inStock(false); // did we see this simTrackId before?
		
		SiPixelCluster::PixelPos newPixelPos(pixel_coord.first, pixel_coord.second); // coordinates to the pixel
		
		//loop over the pixels from the cluster to get the charge in this pixel
		int newPixelCharge(0); //fraction times charge in the original cluster pixel
		
		const std::vector<SiPixelCluster::Pixel>& pixvector = (*c.cluster).pixels();
		
		for(std::vector<SiPixelCluster::Pixel>::const_iterator itPix = pixvector.begin(); itPix != pixvector.end(); itPix++)
		  {
		    if (((int) itPix->x) == ((int) pixel_coord.first)&&(((int) itPix->y) == ((int) pixel_coord.second)))
		      {
			newPixelCharge = (int) (linkiter->fraction()*itPix->adc); 
		      }
		  }
		
		if ( newPixelCharge < 2500 ) 
		  continue; 
		
		//add the pixel to an already existing cluster if the charge is above the threshold
		int clusVecPos = 0;
		std::vector<int>::const_iterator sTIter =  simTrackIdV.begin();
	      
		for ( ; sTIter < simTrackIdV.end(); sTIter++) 
		  {
		    if (((*sTIter)== (int) linkiter->SimTrackId())) 
		      {
		      inStock=true; // now we saw this id before
		      // 	  //		  std::cout << " adding a pixel to the cluster " << (int) (clusVecPos) <<std::endl;
		      // 	  //		    std::cout << "newPixelCharge " << newPixelCharge << std::endl;
		      splittedCluster.at(clusVecPos).add(newPixelPos,newPixelCharge); // add the pixel to the cluster
		      }
		    clusVecPos++;
		  }
		
	      //look if the splitted cluster was already made before, if not create one
		
		if ( !inStock ) 
		  {
		    //		std::cout << "creating a new cluster " << std::endl;
		    simTrackIdV.push_back(linkiter->SimTrackId()); // add the track id to the vector
		  splittedCluster.push_back(SiPixelCluster(newPixelPos,newPixelCharge)); // add the cluster to the vector
		  }
	      }
	  }
	
	//    std::cout << "will add clusters : simTrackIdV.size() " << simTrackIdV.size() << std::endl;
      
	if ( ( ( (int)simTrackIdV.size() ) == 1 ) || ( *c.cluster).size()==1 ) 
	  { 
	    //	    cout << "putting in this cluster" << endl;
	    output.push_back(*c.cluster );
	    //      std::cout << "cluster added " << output.size() << std::endl;
	  }
	else 
	{  	  
	  for (std::vector<SiPixelCluster>::const_iterator cIter = splittedCluster.begin(); cIter != splittedCluster.end(); cIter++ )
	    {
	      output.push_back( (*cIter) );   
	    }
	}
	
	simTrackIdV.clear();  
	splittedCluster.clear();
      }//if (isearch != pixeldigisimlink->end())
    }
  else if ( tmpSplitPixel_ )
    { 
      bool cluster_was_successfully_split = false;
      
      const SiPixelCluster* thePixelCluster = static_cast<const SiPixelCluster*>(c.cluster);

      if ( thePixelCluster )
	{ 
	  // Do not attempt to split clusters of size one
	  if ( (int)thePixelCluster->size() <= 1 )
	    {    
	      // Do nothing.
	      //cout << "Will not attempt to split this clusters: " << endl;
	      //cout << "(int)thePixelCluster->size() = " << (int)thePixelCluster->size() << endl;
	    }
	  else
	    {
	      // For barrel use template id 40 and for endcaps use template id 41
	      int ID = -99999;
	      if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelBarrel  )
		{
		  //		  cout << "We are in the barrel : " << (int)PixelSubdetector::PixelBarrel << endl;
		  ID = 40;
		}
	      else if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelEndcap )
		{
		  //		  cout << "We are in the endcap : " << (int)PixelSubdetector::PixelEndcap << endl;
		  ID = 41;
		}
	      else 
		{
		  // cout << "Not a pixel detector !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
		}
	      

	      // Begin: determine incident angles ============================================================

	      float cotalpha_ = -99999.9;
	      float cotbeta_  = -99999.9;

	      // First, determine track angles from straight track approximation
 
	      // Find crude cluster center. 
	      float xcenter = thePixelCluster->x();
	      float ycenter = thePixelCluster->y();
	      
	      const GeomDetUnit* theDet = geometry_->idToDetUnit( detId );
	      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>( theDet );
	      
	      const PixelTopology* theTopol = (&(pixDet->specificTopology()));
	    
	      // Transform from measurement to local coordinates (in cm)
	      LocalPoint lp = theTopol->localPosition( MeasurementPoint(xcenter, ycenter) );
	      
	      // Transform from local to global coordinates
	      GlobalPoint gp0 = theDet->surface().toGlobal( lp );
	      
	      // Make a vector pointing from the PV to the cluster center 
	      GlobalPoint gp(gp0.x()-vtx.x(), gp0.y()-vtx.y(), gp0.z()-vtx.z() );
	      
	      // Make gp a unit vector, gv, pointing from the PV to the cluster center
	      float gp_mod = sqrt( gp.x()*gp.x() + gp.y()*gp.y() + gp.z()*gp.z() );
	      float gpx = gp.x()/gp_mod;
	      float gpy = gp.y()/gp_mod;
	      float gpz = gp.z()/gp_mod;
	      GlobalVector gv(gpx, gpy, gpz);

	      // Make unit vectors in local coordinates and then transform them in global coordinates
	      const Local3DVector lvx(1.0, 0.0, 0.0);
	      GlobalVector gvx = theDet->surface().toGlobal( lvx );
	      const Local3DVector lvy(0.0, 1.0, 0.0);
	      GlobalVector gvy = theDet->surface().toGlobal( lvy );
	      const Local3DVector lvz(0.0, 0.0, 1.0);
	      GlobalVector gvz = theDet->surface().toGlobal( lvz );
	      
	      // Calculate angles alpha and beta
	      float gv_dot_gvx = gv.x()*gvx.x() + gv.y()*gvx.y() + gv.z()*gvx.z();
	      float gv_dot_gvy = gv.x()*gvy.x() + gv.y()*gvy.y() + gv.z()*gvy.z();
	      float gv_dot_gvz = gv.x()*gvz.x() + gv.y()*gvz.y() + gv.z()*gvz.z();
	      
	      //float alpha_ = atan2( gv_dot_gvz, gv_dot_gvx );
	      //float beta_  = atan2( gv_dot_gvz, gv_dot_gvy );
	      
	      cotalpha_ = gv_dot_gvx / gv_dot_gvz;
	      cotbeta_  = gv_dot_gvy / gv_dot_gvz;
	      
	      


	      // Attempt to get a better angle from tracks (either pixel tracks or full tracks)
	      if ( !useStraightTracks_ )
		{
		  // Use either pixel tracks (useTrajectories_ = False) or fully reconstructed tracks (useTrajectories_ = True)
		  // When pixel/full tracks are not associated with the current cluster, will use angles from straight tracks
		  
		  // These are the tracks associated with this cluster
		  std::vector<TrackAndState> vec_tracks_states = c.tracks;
		  
		  

		  if ( (int)vec_tracks_states.size() > 0 )
		    {
		      //cout << "There is at least one track associated with this cluster. Pick the one with largest Pt." << endl;
		      //cout << "(int)vec_tracks_states.size() = " << (int)vec_tracks_states.size() << endl;
		      
		      int index_max_pt = -99999;  // index of the track with the highest Pt
		      float max_pt = -99999.9;
		      
		      for (int i=0; i<(int)vec_tracks_states.size(); ++i )
			{
			  const reco::Track* one_track = vec_tracks_states[i].track;
			  
			  if ( one_track->pt() > max_pt )
			    {
			      index_max_pt = i;
			      max_pt = one_track->pt();
			    }  
			}
		      
		      // Pick the tsos from the track with highest Pt
		      // gavril: Should we use highest Pt or best Chi2 ?
		      TrajectoryStateOnSurface one_tsos = vec_tracks_states[index_max_pt].state;
		      
		      LocalTrajectoryParameters ltp = one_tsos.localParameters();
		      
		      LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
		      
		      float locx = localDir.x();
		      float locy = localDir.y();
		      float locz = localDir.z();
		      
		      //alpha_ = atan2( locz, locx );
		      //beta_  = atan2( locz, locy );
		      
		      cotalpha_ = locx/locz;
		      cotbeta_  = locy/locz;
		      
		     
		      
		    } // if ( (int)vec_tracks_states.size() > 0 )
		      
		} // if ( !useStraightTracks_ )

	      // End: determine incident angles ============================================================


	      
	      //cout << "Calling qbin to see if the cluster has to be split..." << endl;	      
              SiPixelTemplate templ_(thePixelTemp_);
              SiPixelTemplate2D templ2D_(thePixelTemp2D_);
	      int templQbin_ = templ_.qbin( ID, cotalpha_, cotbeta_, thePixelCluster->charge() );

	      if ( templQbin_ < 0 || templQbin_ > 5  )
		{
		  // gavril : check this....
		  // cout << "Template call failed. Cannot decide if cluster should be split !!!!!!! " << endl;
		  // cout << "Do nothing." << endl;
		}
	      else // if ( templQbin_ < 0 || templQbin_ > 5  ) {...} else
                {
		  //cout << " Returned OK from PixelTempReco2D..." << endl;
		  
		  // Check for split clusters: we split the clusters with larger than expected charge: templQbin_ == 0 
		  if ( templQbin_ != 0 ) 
		    {
		      // gavril: do not split this cluster
		      //cout << "TEMPLATE SPLITTER SAYS : NO SPLIT " << endl;
		      //cout << "This cluster will note be split !!!!!!!!!! " << endl; 
		    }
		  else // if ( templQbin_ != 0 ) {...} else
		    {
		      //cout << "TEMPLATE SPLITTER SAYS : SPLIT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		      //cout << "Found a cluster that has to be split. templQbin_ = " << templQbin_ << endl;
		      
		      // gavril: Call the template splitter
		      //cout << "Calling the splitter..." << endl;
		     
		      // Put the pixels of this clusters in a 2x2 array to be used by the template splitter
		      
		      const std::vector<SiPixelCluster::Pixel> & pixVec = thePixelCluster->pixels();
		      std::vector<SiPixelCluster::Pixel>::const_iterator pixIter = pixVec.begin(), pixEnd = pixVec.end();
		      
		      const int cluster_matrix_size_x = 13;
		      const int cluster_matrix_size_y = 21;
		      
		      boost::multi_array<float, 2> clust_array_2d(boost::extents[cluster_matrix_size_x][cluster_matrix_size_y]);
		      
		      int row_offset = thePixelCluster->minPixelRow();
		      int col_offset = thePixelCluster->minPixelCol();
		      
		      // Copy clust's pixels (calibrated in electrons) into clust_array_2d;
	      
		      for ( ; pixIter != pixEnd; ++pixIter ) 
			{
			  int irow = int(pixIter->x) - row_offset;   // do we need +0.5 ???
			  int icol = int(pixIter->y) - col_offset;   // do we need +0.5 ???
			  
			  if ( irow<cluster_matrix_size_x && icol<cluster_matrix_size_y )
			    {
			      clust_array_2d[irow][icol] = (float)pixIter->adc;
			    }
			}
		      
		      // Make and fill the bool arrays flagging double pixels
		      std::vector<bool> ydouble(cluster_matrix_size_y), xdouble(cluster_matrix_size_x);
		      
		      // x directions (shorter), rows
		      for (int irow = 0; irow < cluster_matrix_size_x; ++irow)
			{
			  xdouble[irow] = theTopol->isItBigPixelInX( irow+row_offset );
			}
		      
		      // y directions (longer), columns
		      for (int icol = 0; icol < cluster_matrix_size_y; ++icol) 
			{
			  ydouble[icol] = theTopol->isItBigPixelInY( icol+col_offset );
			}

		      // gavril: Initialize values coming back from SiPixelTemplateSplit::PixelTempSplit
		      float templYrec1_  = -99999.9;
		      float templYrec2_  = -99999.9;
		      float templSigmaY_ = -99999.9;
		      float templProbY_  = -99999.9;
		      float templXrec1_  = -99999.9;
		      float templXrec2_  = -99999.9;
		      float templSigmaX_ = -99999.9;
		      float templProbX_  = -99999.9;
		      float dchisq       = -99999.9;
		      float templProbQ_  = -99999.9;

		      int ierr =
			SiPixelTemplateSplit::PixelTempSplit( ID, 
							      cotalpha_, cotbeta_,
							      clust_array_2d, 
							      ydouble, xdouble,
							      templ_,
							      templYrec1_, templYrec2_, templSigmaY_, templProbY_,
							      templXrec1_, templXrec2_, templSigmaX_, templProbX_,
							      templQbin_, 
							      templProbQ_, 
							      true, 
							      dchisq, 
							      templ2D_ );
		        
		      if ( ierr != 0 )
			{
			  // cout << "Cluster splitting failed: ierr = " << ierr << endl;
			}
		      else
			{
			  // gavril: Cluster was successfully split. 
			  // gavril: Make the two split clusters and put them in the split cluster collection
			  //cout << "Cluster splitting OK: ierr = " << ierr << endl;
			  
			  // 2D templates have origin at the lower left corner of template2d[1][1] which is 
			  // also 2 pixels larger than the cluster container
			  
			  float xsize = templ_.xsize();  // this is the pixel x-size in microns
			  float ysize = templ_.ysize();  // this is the pixel y-size in microns
			  
			  // Shift the coordinates to the 2-D template system			  
			  float yrecp1 = -99999.9;
			  float yrecp2 = -99999.9;
			  float xrecp1 = -99999.9;
			  float xrecp2 = -99999.9;
			  
			  if ( ydouble[0] ) 
			    {				
			      yrecp1 = templYrec1_ + ysize;
			      yrecp2 = templYrec2_ + ysize;
			    } 
			  else 
			    {
			      yrecp1 = templYrec1_ + ysize/2.0;
			      yrecp2 = templYrec2_ + ysize/2.0;
			    }
			  
			  if ( xdouble[0] ) 
			    {
			      xrecp1 = templXrec1_ + xsize;
			      xrecp2 = templXrec2_ + xsize;
			    } 
			  else 
			    {
			      xrecp1 = templXrec1_ + xsize/2.0;
			      xrecp2 = templXrec2_ + xsize/2.0;
			    }
			  
			  //  The xytemp method adds charge to a zeroed buffer
			  
			  float template2d1[BXM2][BYM2];
			  float template2d2[BXM2][BYM2];
			  
			  for ( int j=0; j<BXM2; ++j ) 
			    {
			      for ( int i=0; i<BYM2; ++i ) 
				{
				  template2d1[j][i] = 0.0;
				  template2d2[j][i] = 0.0;
				}
			    }
			   
   
			  bool template_OK 
			    = templ2D_.xytemp(ID, cotalpha_, cotbeta_, 
					      xrecp1, yrecp1, 
					      ydouble, xdouble, 
					      template2d1);
			  
			  template_OK 
			    = template_OK && 
			    templ2D_.xytemp(ID, cotalpha_, cotbeta_, 
					    xrecp2, yrecp2, 
					    ydouble, xdouble, 
					    template2d2);
			  
			  if ( !template_OK ) 
			    {
			      // gavril: check this
			      //cout << "Template is not OK. Fill out with zeros !!!!!!!!!!!!!!! " << endl; 
			      
			      for ( int j=0; j<BXM2; ++j ) 
				{
				  for ( int i=0; i<BYM2; ++i ) 
				    {
				      template2d1[j][i] = 0.0;
				      template2d2[j][i] = 0.0;
				    }
				}
			      
			      if ( !templ_.simpletemplate2D(xrecp1, yrecp1, 
							    xdouble, ydouble, 
							    template2d1) ) 
				{
				  cluster_was_successfully_split = false;
				}
			      
			      if ( !templ_.simpletemplate2D(xrecp2, yrecp2, 
							    xdouble, ydouble, 
							    template2d2) ) 
				{
				  cluster_was_successfully_split = false;
				}
			      
			    } // if ( !template_OK ) 
			  else
			    {
			      cluster_was_successfully_split = true;
			      
			      // Next, copy the 2-d templates into cluster containers, replace small signals with zero.
			      // Cluster 1 and cluster 2 should line up with clust_array_2d (same origin and pixel indexing)
			      
			      float q50 = templ_.s50();
			      
			      
			      float cluster1[TXSIZE][TYSIZE];
			      float cluster2[TXSIZE][TYSIZE];  //Note that TXSIZE = BXM2 - 2, TYSIZE = BYM2 - 2
			      
			      for ( int j=0; j<TXSIZE; ++j ) 
				{
				  for ( int i=0; i<TYSIZE; ++i ) 
				    {
				      cluster1[j][i] = template2d1[j+1][i+1];
				      
				      if ( cluster1[j][i] < q50 ) 
					cluster1[j][i] = 0.0;
				      
				      cluster2[j][i] = template2d2[j+1][i+1];
				      
				      if ( cluster2[j][i] < q50 ) 
					cluster2[j][i] = 0.0;

				      //cout << "cluster1[j][i] = " << cluster1[j][i] << endl;
				      //cout << "cluster2[j][i] = " << cluster2[j][i] << endl;
				    }
				}
				
			  
			      // Find the coordinates of first pixel in each of the two split clusters 
			      int i1 = -999;
			      int j1 = -999;
			      int i2 = -999;
			      int j2 = -999;
			     			     
			      bool done_searching = false; 
			      for ( int i=0; i<13 && !done_searching; ++i)
				{
				  for (int j=0; j<21 && !done_searching; ++j)
				    {
				      if ( cluster1[i][j] > 0 )
					{
					  i1 = i;
					  j1 = j;
					  done_searching = true;
					}    
				    }
				}

			      done_searching = false; 
			      for ( int i=0; i<13 && !done_searching; ++i)
				{
				  for (int j=0; j<21 && !done_searching; ++j)
				    {
				      if ( cluster2[i][j] > 0 )
					{
					  i2 = i;
					  j2 = j;
					  done_searching = true;
					}
				    }
				}
			      
			      
			      // Make clusters out of the first pixels in each of the two split clsuters

			      SiPixelCluster cl1( SiPixelCluster::PixelPos( i1 + row_offset, j1 + col_offset), 
						  cluster1[i1][j1] );
			      
			      SiPixelCluster cl2( SiPixelCluster::PixelPos( i2 + row_offset, j2 + col_offset), 
						  cluster2[i2][j2] );
			      

			      // Now add the rest of the pixels to the split clusters

			      for ( int i=0; i<13; ++i)
				{
				  for (int j=0; j<21; ++j)
				    {
				      
				      if ( cluster1[i][j] > 0.0 && (i!=i1 || j!=j1 ) )
					{
					  cl1.add( SiPixelCluster::PixelPos( i + row_offset, j + col_offset), 
						   cluster1[i][j] ); 

					  //cout << "cluster1[i][j] = " << cluster1[i][j] << endl;
					}
				      
				      
				      if ( cluster2[i][j] > 0.0 && (i!=i2 || j!=j2 ) )
					{						      
					  cl2.add( SiPixelCluster::PixelPos( i + row_offset, j + col_offset), 
						   cluster2[i][j] );
					  
					  //cout << "cluster2[i][j] = " << cluster2[i][j] << endl;
					}
				    }
				}
			      
			      // Attach errors and probabilities to the split Clusters
			      // The errors will be later attahed to the SiPixelRecHit


			      cl1.setSplitClusterErrorX( templSigmaX_ );
			      cl1.setSplitClusterErrorY( templSigmaY_ );
			      //cl1.prob_q = templProbQ_;
			      
			      cl2.setSplitClusterErrorX( templSigmaX_ );
			      cl2.setSplitClusterErrorY( templSigmaY_ );
			      //cl2.prob_q = templProbQ_;


			      // Save the split clusters
			      output.push_back( cl1 );
			      output.push_back( cl2 );     
			     
			      // Some sanity checks

			      if ( (int)cl1.size() <= 0 )
				{
				  edm::LogError("TrackClusterSplitter : ") 
				    << "1) Cluster of size = " << (int)cl1.size() << " !!! " << endl;
				}

			      if ( (int)cl2.size() <= 0 )
				{
				  edm::LogError("TrackClusterSplitter : ") 
				    << "2) Cluster of size = " << (int)cl2.size() << " !!! " << endl;
				}

			      if ( cl1.charge() <= 0 )
				{
				  edm::LogError("TrackClusterSplitter : ") 
				    << "1) Cluster of charge = " << (int)cl1.charge() << " !!! " << endl;
				  
				}

			      if ( cl2.charge() <= 0 )
				{
				  edm::LogError("TrackClusterSplitter : ") 
				    << "2) Cluster of charge = " << (int)cl2.charge() << " !!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
				}
				  
			     
			    } // if ( !template_OK ) ... else 
			  
			} // if ( ierr2 != 0 ) ... else
		      
		    } // if ( templQbin_ != 0 ) ... else
		  
		} // else // if ( templQbin_ < 0 || templQbin_ > 5  ) {...} else
	      
	    } // if ( (int)thePixelCluster->size() <= 1 ) {... } else
	  
	  if ( !cluster_was_successfully_split )
	    output.push_back(*c.cluster);
	  
	} // if ( theSiPixelCluster )
      else 
	{
	  throw cms::Exception("TrackClusterSplitter :") 
	    << "This is not a SiPixelCluster !!! " << "\n"; 
	}
    }
  else 
    {
      // gavril : Neither sim nor template splitter. Just add the cluster as it was.
      // original from G.P.
      output.push_back( *c.cluster );
    }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackClusterSplitter);
