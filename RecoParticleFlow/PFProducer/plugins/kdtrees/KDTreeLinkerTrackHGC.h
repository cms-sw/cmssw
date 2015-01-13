#ifndef KDTreeLinkerTrackHGC_h
#define KDTreeLinkerTrackHGC_h

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TMath.h"

// This class is used to find all links between Tracks and ECAL clusters
// using a KDTree algorithm.
// It is used in PFBlockAlgo.cc in the function links().
template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
class KDTreeLinkerTrackHGC : public KDTreeLinkerBase
{
 public:
  KDTreeLinkerTrackHGC();
  ~KDTreeLinkerTrackHGC();
  
  // With this method, we create the list of psCluster that we want to link.
  void insertTargetElt(reco::PFBlockElement		*track);

  // Here, we create the list of ecalCluster that we want to link. From ecalCluster
  // and fraction, we will create a second list of rechits that will be used to
  // build the KDTree.
  void insertFieldClusterElt(reco::PFBlockElement	*hgcCluster);  

  // The KDTree building from rechits list.
  void buildTree();
  
  // Here we will iterate over all tracks. For each track intersection point with ECAL, 
  // we will search the closest rechits in the KDTree, from rechits we will find the 
  // ecalClusters and after that we will check the links between the track and 
  // all closest ecalClusters.  
  void searchLinks();
    
  // Here, we will store all PS/ECAL founded links in the PFBlockElement class
  // of each psCluster in the PFmultilinks field.
  void updatePFBlockEltWithLinks();
  
  // Here we free all allocated structures.
  void clear();
 
 private:
  // Data used by the KDTree algorithm : sets of Tracks and ECAL clusters.
  BlockEltSet		targetSet_;
  BlockEltSet		fieldClusterSet_;

  // Sets of rechits that compose the ECAL clusters. 
  RecHitSet		rechitsSet_;
  
  // Map of linked Track/ECAL clusters.
  BlockElt2BlockEltMap	target2ClusterLinks_;

  // Map of the ECAL clusters associated to a rechit.
  RecHit2BlockEltMap	rechit2ClusterLinks_;
    
  // KD trees
  KDTreeLinkerAlgo	tree_;

};

template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
  KDTreeLinkerTrackHGC<the_layer,RHscaling>::KDTreeLinkerTrackHGC()
  : KDTreeLinkerBase()
{}

template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
KDTreeLinkerTrackHGC<the_layer,RHscaling>::~KDTreeLinkerTrackHGC()
{
  clear();
}


template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
void
KDTreeLinkerTrackHGC<the_layer,RHscaling>::insertTargetElt(reco::PFBlockElement	*track)
{
  // for HGC we need to iterate through each compartment and check the linking there
  for( unsigned ilay = (unsigned)reco::PFTrajectoryPoint::HGC_ECALEntrance; ilay <= (unsigned)the_layer; ++ilay ) {
    if( track->trackRefPF()->extrapolatedPoint( (reco::PFTrajectoryPoint::LayerType)ilay ).isValid() ) { 
      //std::cout << *(track->trackRefPF()) << std::endl;
      targetSet_.insert(track);
    }
  }
}

template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
void
KDTreeLinkerTrackHGC<the_layer,RHscaling>::insertFieldClusterElt(reco::PFBlockElement	*hgcCluster)
{
  reco::PFClusterRef clusterref = hgcCluster->clusterRef();
  //std::cout << "cluster inserted with position: " << clusterref->position() 
  //    << ' ' << clusterref->positionREP() << std::endl;

  // This test is more or less done in PFBlockAlgo.h. In others cases, it should be switch on.
  //   if (!((clusterref->layer() == PFLayer::ECAL_ENDCAP) ||
  // 	(clusterref->layer() == PFLayer::ECAL_BARREL)))
  //     return;

  const std::vector<reco::PFRecHitFraction> &fraction = clusterref->recHitFractions();
  //std::cout << "   cluster has " << fraction.size() << " rechits!" << std::endl;
  
  // We create a list of hgcCluster
  fieldClusterSet_.insert(hgcCluster);
  /*
  DetId seedId( clusterref->seed() );
  unsigned seedLayer = 1000;  
  if( seedId.det() == DetId::Forward ) {
    if( seedId.subdetId() == HGCEE ) {
      seedLayer = HGCEEDetId(seedId).layer();
    } else {
      seedLayer = HGCHEDetId(seedId).layer();
    }
  } else {
    throw cms::Exception("BadSeedRecHit") 
      << "HGC KDTree Linker only accepts HGC DetIds! got: " << seedId.det();
  }
  */
  
  for(size_t rhit = 0; rhit < fraction.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fraction[rhit].recHitRef();
    double fract = fraction[rhit].fraction();
    
    /*
    DetId rhId( rh->detId() );
    unsigned rhLayer = 1000;

    if( rhId.det() == DetId::Forward ) {
      if( rhId.subdetId() == HGCEE ) {
	rhLayer = HGCEEDetId(rhId).layer();
      } else {
	rhLayer = HGCHEDetId(rhId).layer();
      }
    } else {
      throw cms::Exception("BadRecHit") 
	<< "HGC KDTree Linker only accepts HGC DetIds! got: " << rhId.det();
    }
    */
    
    if ( (rh.isNull()) || (fract < 1E-4) )
      continue;
      
    const reco::PFRecHit& rechit = *rh;
    
    
    //std::cout << "added rechit in layer " << the_layer << ' ' << rechit.position() << ' ' << rechit.positionREP() << std::endl;
    /*
    const std::vector< math::XYZPoint >& cornersxyz = rechit.getCornersXYZ();
    for( unsigned i = 0 ; i < cornersxyz.size(); ++i ) {
      std::cout << "\t corner : " << i << " : " << cornersxyz[i] << std::endl;
    }
    */
    
    

    // We save the links rechit to EcalClusters
    rechit2ClusterLinks_[&rechit].insert(hgcCluster);
    
    // We create a liste of rechits
    rechitsSet_.insert(&rechit);
  }
}

template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
void 
  KDTreeLinkerTrackHGC<the_layer,RHscaling>::buildTree()
{
  // List of pseudo-rechits that will be used to create the KDTree
  std::vector<KDTreeNodeInfo> eltList;

  // Filling of this list
  for(RecHitSet::const_iterator it = rechitsSet_.begin(); 
      it != rechitsSet_.end(); it++) {
    
    const reco::PFRecHit::REPPoint &posrep = (*it)->positionREP();
    
    KDTreeNodeInfo rh1 (*it, posrep.Eta(), posrep.Phi());
    eltList.push_back(rh1);
    
    // Here we solve the problem of phi circular set by duplicating some rechits
    // too close to -Pi (or to Pi) and adding (substracting) to them 2 * Pi.
    if (rh1.dim2 > (M_PI)) {
      double phi = rh1.dim2 - 2 * M_PI;
      KDTreeNodeInfo rh2(*it, posrep.Eta(), phi); 
      eltList.push_back(rh2);
    }

    if (rh1.dim2 < (M_PI * -1.0)) {
      double phi = rh1.dim2 + 2 * M_PI;
      KDTreeNodeInfo rh3(*it, posrep.Eta(), phi); 
      eltList.push_back(rh3);
    }
  }

  // Here we define the upper/lower bounds of the 2D space (eta/phi).
  double phimin = -1.0 * M_PI ;
  double phimax = M_PI ;

  // etamin-etamax, phimin-phimax
  KDTreeBox region(-3.0, 3.0, phimin, phimax);

  // We may now build the KDTree
  tree_.build(eltList, region);
}

template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
void
  KDTreeLinkerTrackHGC<the_layer,RHscaling>::searchLinks()
{
  // Must of the code has been taken from LinkByRecHit.cc
  //std::cout << "running searchLinks() for : " << the_layer 
  //<< ' ' << targetSet_.size() << std::endl;

  // We iterate over the tracks.
  for(BlockEltSet::iterator it = targetSet_.begin(); 
      it != targetSet_.end(); it++) {
	
    reco::PFRecTrackRef trackref = (*it)->trackRefPF();

    // We set the multilinks flag of the track to true. It will allow us to 
    // use in an optimized way our algo results in the recursive linking algo.
    (*it)->setIsValidMultilinks(true);

    // for HGC we need to iterate through each compartment and check the linking there
    for( unsigned ilay = (unsigned)reco::PFTrajectoryPoint::HGC_ECALEntrance; ilay <= (unsigned)the_layer; ++ilay ) {
    
      const reco::PFTrajectoryPoint& atECAL = 
	trackref->extrapolatedPoint((reco::PFTrajectoryPoint::LayerType)ilay);

      //std::cout << the_layer << " track position at " << ilay << " surface: " << atECAL.position() << std::endl;

      // The track didn't reach ecal
      if( ! atECAL.isValid() ) {
	//std::cout << "extrapolation to " << the_layer << " was bad!" << std::endl;
	continue;
      }
      
      const reco::PFTrajectoryPoint& atVertex = 
	trackref->extrapolatedPoint( reco::PFTrajectoryPoint::ClosestApproach );
      
      double trackPt = sqrt(atVertex.momentum().Vect().Perp2());
      double tracketa = atECAL.positionREP().Eta();
      double trackphi = atECAL.positionREP().Phi();
      
      //double trackx = atECAL.position().X();
      //double tracky = atECAL.position().Y();
      //double trackz = atECAL.position().Z();    
      
      // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
      // Same envelope for cap et barrel rechits.
      double range = getCristalPhiEtaMaxSize() * (2.0 + 1.0 / std::min(1., 0.5*trackPt )); 
      //std::cout << range << std::endl;
      
      // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
      std::vector<KDTreeNodeInfo> recHits;
      KDTreeBox trackBox(tracketa-range, tracketa+range, trackphi-range, trackphi+range);
      tree_.search(trackBox, recHits);
      /*
	std::cout << "track pt: " << trackPt << ' ' << tracketa << ' ' << trackphi << std::endl;
	std::cout << "got " << recHits.size() << " rechits from the KDtree search." << std::endl;
      */
      // Here we check all rechit candidates using the non-approximated method.
      for(std::vector<KDTreeNodeInfo>::const_iterator rhit = recHits.begin(); 
	  rhit != recHits.end(); ++rhit) {
	
	//const std::vector< math::XYZPoint >& cornersxyz      = rhit->ptr->getCornersXYZ();
	const math::XYZPoint& posxyz			   = rhit->ptr->position();
	const reco::PFRecHit::REPPoint &rhrep		   = rhit->ptr->positionREP();
	const std::vector<reco::PFRecHit::REPPoint>& corners = rhit->ptr->getCornersREP();
	//const auto& corners_xyz = rhit->ptr->getCornersXYZ();
	//if(cornersxyz.size() != 4) continue;
	
	
	//double rhsizeEta = fabs(corners[0].Eta() - corners[2].Eta());
	double rhsizePhi = fabs(corners[0].Phi() - corners[2].Phi());
	if ( rhsizePhi > M_PI ) rhsizePhi = 2.*M_PI - rhsizePhi;
	
	//double deta = fabs(rhrep.Eta() - tracketa);
	double dphi = fabs(rhrep.Phi() - trackphi);
	if ( dphi > M_PI ) dphi = 2.*M_PI - dphi;
	/*
	  std::cout << the_layer << " rhsize eta/phi: " << rhsizeEta << '/' << rhsizePhi
	  << " deta/dphi: " << deta << '/' << dphi << " z: " 
	  << rhit->ptr->position().z() << '/' << atECAL.position().z() << std::endl;
	*/
	
	// Find all clusters associated to given rechit
	RecHit2BlockEltMap::iterator ret = rechit2ClusterLinks_.find(rhit->ptr);
	
	for(BlockEltSet::const_iterator clusterIt = ret->second.begin(); 
	    clusterIt != ret->second.end(); clusterIt++) {
	
	  reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
	  
	  //std::cout << " cluster pt: " << clusterref->energy()/std::cosh(clusterref->position().eta()) << std::endl;
	  
	  //double clusterz = clusterref->position().Z();
	  //int fracsNbr = clusterref->recHitFractions().size();
	  /*
	  double x[5];
	  double y[5];
	  for ( unsigned jc=0; jc<4; ++jc ) {
	    math::XYZPoint cornerposxyz = cornersxyz[jc];
	    x[jc] = cornerposxyz.X() + (cornerposxyz.X()-posxyz.X())
	      * (2.00+RHscaling/(0.1*trackPt));
	    y[jc] = cornerposxyz.Y() + (cornerposxyz.Y()-posxyz.Y())
	      * (2.00+RHscaling/(0.1*trackPt));
	    //std::cout << "hit corner x/y/z: " << cornerposxyz << std::endl;
	  }
	  */
	  
	  /*
	    std::cout << "hit position x/y/z: " << posxyz
	    << ' ' <<  rhrep << std::endl;
	    std::cout << "tk position  x/y  : " << trackx << ' ' << tracky
	    << ' ' << tracketa << ' ' << trackphi << std::endl;
	  */
	    
	  /*
	  x[4] = x[0];
	  y[4] = y[0];
	  */
	  
	  // Check if the track and the cluster are linked
	  //std::cout << "inside linker: " << atECAL.position() << ' ' << posxyz << std::endl;
	  if( (atECAL.position() - posxyz).rho() < 7.0 ) {
	    target2ClusterLinks_[*it].insert(*clusterIt);	  
	  }
	}
      }
    }
  }
}


template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
void
KDTreeLinkerTrackHGC<the_layer,RHscaling>::updatePFBlockEltWithLinks()
{
  //TODO YG : Check if cluster positionREP() is valid ?

  // Here we save in each track the list of phi/eta values of linked clusters.
  for (BlockElt2BlockEltMap::iterator it = target2ClusterLinks_.begin();
       it != target2ClusterLinks_.end(); ++it) {
    reco::PFMultiLinksTC multitracks(true);

    for (BlockEltSet::iterator jt = it->second.begin();
	 jt != it->second.end(); ++jt) {

      double clusterPhi = (*jt)->clusterRef()->positionREP().Phi();
      double clusterEta = (*jt)->clusterRef()->positionREP().Eta();

      multitracks.linkedClusters.push_back(std::make_pair(clusterPhi, clusterEta));
    }

    it->first->setMultilinks(multitracks);
  }
}

template<reco::PFTrajectoryPoint::LayerType the_layer,unsigned RHscaling>
void
KDTreeLinkerTrackHGC<the_layer,RHscaling>::clear()
{
  targetSet_.clear();
  fieldClusterSet_.clear();

  rechitsSet_.clear();

  rechit2ClusterLinks_.clear();
  target2ClusterLinks_.clear();

  tree_.clear();
}

#endif /* !KDTreeLinkerTrackHGC_h */
