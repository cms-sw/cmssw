#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerPSEcal.h"

#include "TMath.h"

using namespace KDTreeLinker;

KDTreeLinkerPSEcal::KDTreeLinkerPSEcal(double	phiOffset,
				       double ecalDiameter,
				       bool debug)
  : phiOffset_ (phiOffset), 
    ecalDiameter_(ecalDiameter),
    debug_(debug),
    resPSpitch_ (0.19),
    resPSlength_ (6.1),
    ps1ToEcal_ (1.072),
    ps2ToEcal_ (1.057)
{}

KDTreeLinkerPSEcal::~KDTreeLinkerPSEcal()
{
  clear();
}

void
KDTreeLinkerPSEcal::setPhiOffset(double phiOffset)
{
  phiOffset_ = phiOffset;
}

double
KDTreeLinkerPSEcal::getPhiOffset() const
{
  return phiOffset_;
}

void
KDTreeLinkerPSEcal::setEcalDiameter(double ecalDiameter)
{
  ecalDiameter_ = ecalDiameter;
}

double
KDTreeLinkerPSEcal::getEcalDiameter() const
{
  return ecalDiameter_;
}

void 
KDTreeLinkerPSEcal::setDebug(bool isDebug)
{
  debug_ = isDebug;
}

void
KDTreeLinkerPSEcal::insertPS(reco::PFBlockElement* psCluster)
{
  psSet_.insert(psCluster);
}


void
KDTreeLinkerPSEcal::insertEcal(const reco::PFBlockElement*			ecalCluster,
				     const std::vector<reco::PFRecHitFraction>& fraction)
{
  reco::PFClusterRef clusterref = ecalCluster->clusterRef();
  if (clusterref->layer() != PFLayer::ECAL_ENDCAP)
    return;
	  
  // We create a list of cluster
  ecalSet_.insert(ecalCluster);

  double clusterz = clusterref->position().Z();
  RecHitSet& rechitsSet = (clusterz < 0) ? rechitsNegSet_ : rechitsPosSet_;

  for(size_t rhit = 0; rhit < fraction.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fraction[rhit].recHitRef();
    double fract = fraction[rhit].fraction();

    if ((rh.isNull()) || (fract < 1E-4))
      continue;
      
    const reco::PFRecHit& rechit = *rh;
      
    // We save the links rechit to Clusters
    rechitEcalLinks_[&rechit].insert(ecalCluster);
    
    // We create a liste of rechits
    rechitsSet.insert(&rechit);
  }
}

void 
KDTreeLinkerPSEcal::buildTree()
{
  buildTree(rechitsNegSet_, treeNeg_);
  buildTree(rechitsPosSet_, treePos_);
}

void 
KDTreeLinkerPSEcal::buildTree(const RecHitSet&	rechitsSet,
			      KDTreeLinkerAlgo&	tree)
{
    // List of pseudo-rechits that will be used to create the KDTree
  std::vector<RHinfo> eltList;

  // Filling of this list
  for(RecHitSet::const_iterator it = rechitsSet.begin(); 
      it != rechitsSet.end(); it++) {

    const reco::PFRecHit* rh = *it;
    const math::XYZPoint& posxyz = rh->position(); // * zPS / clusterz;
        
    RHinfo rhinfo (rh, posxyz.X(), posxyz.Y());
    eltList.push_back(rhinfo);
  }

  //xmin-xmax, ymain-ymax
  TBox region(-150., 150., -150., 150.);

  // We may now build the KDTree
  tree.build(eltList, region);
}

void
KDTreeLinkerPSEcal::searchLinks()
{
  // Must of the code has been taken from LinkByRecHit.cc

  // We iterate over the tracks.
  for(BlockEltSet::iterator it = psSet_.begin(); 
      it != psSet_.end(); it++) {

    (*it)->setIsValidMultilinks(true);
	
    reco::PFClusterRef clusterPSRef = (*it)->clusterRef();
    const reco::PFCluster& clusterPS = *clusterPSRef;

    // PS cluster position, extrapolated to ECAL
    double zPS = clusterPS.position().Z();
    double xPS = clusterPS.position().X();
    double yPS = clusterPS.position().Y();

    double deltaX = 0.;
    double deltaY = 0.;
    double xPSonEcal = xPS;
    double yPSonEcal = yPS;
    switch (clusterPS.layer()) {
    case PFLayer::PS1:
      // vertical strips, measure x with pitch precision
      deltaX = resPSpitch_;
      deltaY = resPSlength_;
      xPSonEcal *= ps1ToEcal_;
      yPSonEcal *= ps1ToEcal_;
      break;
    case PFLayer::PS2:
      // horizontal strips, measure y with pitch precision
      deltaY = resPSpitch_;
      deltaX = resPSlength_;
      xPSonEcal *= ps2ToEcal_;
      yPSonEcal *= ps2ToEcal_;
      break;
    default:
      ////////////////////////////////////////////////////////////////////////////////
      std::cout << "------------------------- IMPOSSIBLE BECAME POSSIBLE : searchLinks PSEcal"  << std::endl;
      return;
    }
 
    
    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    
    
    double maxEcalRadius = 2.0 * getEcalDiameter() / 2.;

    double rangeX = maxEcalRadius * (1 + (0.05 + 1.0 / maxEcalRadius * deltaX / 2.)); 
    double rangeY = maxEcalRadius * (1 + (0.05 + 1.0 / maxEcalRadius * deltaY / 2.)); 
    
    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<RHinfo> recHits;
    TBox trackBox(xPSonEcal - rangeX, xPSonEcal + rangeX, 
		  yPSonEcal - rangeY, yPSonEcal + rangeY);

    if (zPS < 0)
      treeNeg_.search(trackBox, recHits);
    else
      treePos_.search(trackBox, recHits);


    for(std::vector<RHinfo>::const_iterator rhit = recHits.begin(); 
	rhit != recHits.end(); ++rhit) {
           
      const std::vector< math::XYZPoint >& corners = rhit->ptr->getCornersXYZ();
      if(corners.size() != 4) continue;

      // Find all clusters associated to given rechit
      RecHitClusterMap::iterator ret = rechitEcalLinks_.find(rhit->ptr);
      
      for(BlockEltSet_const::const_iterator clusterIt = ret->second.begin(); 
	  clusterIt != ret->second.end(); clusterIt++) {
	
	reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
	double clusterz = clusterref->position().Z();

	const math::XYZPoint& posxyz = rhit->ptr->position() * zPS / clusterz;

	double x[5];
	double y[5];
	for ( unsigned jc=0; jc<4; ++jc ) {
	  math::XYZPoint cornerpos = corners[jc] * zPS / clusterz;
	  x[jc] = cornerpos.X() + (cornerpos.X()-posxyz.X()) * (0.05 +1.0/fabs((cornerpos.X()-posxyz.X()))*deltaX/2.);
	  y[jc] = cornerpos.Y() + (cornerpos.Y()-posxyz.Y()) * (0.05 +1.0/fabs((cornerpos.Y()-posxyz.Y()))*deltaY/2.);
	}

	x[4] = x[0];
	y[4] = y[0];
	
	bool isinside = TMath::IsInside(xPS,
					yPS,
					5,x,y);
	
	// Check if the track and the cluster are linked
	if( isinside )
	  PSEcalLinks_[*it].insert(*clusterIt);	
      }
    }
    
  }
}

void
KDTreeLinkerPSEcal::updateTracksWithLinks()
{
  //debug_ = true;
  // For debug purpose
  std::vector<int> histo(25, 0);

  //TODO YG : Check if cluster positionREP() is valid

  // Here we save in each track the list of phi/eta values of linked clusters.
  for (BlockEltClusterMap::iterator it = PSEcalLinks_.begin();
       it != PSEcalLinks_.end(); ++it) {
    reco::PFMultiLinksTC multitracks(true);

    if (debug_)
      if ( it->second.size() < 25)
	++(histo[it->second.size()]);

    for (BlockEltSet_const::iterator jt = it->second.begin();
	 jt != it->second.end(); ++jt) {

      double clusterPhi = (*jt)->clusterRef()->positionREP().Phi();
      double clusterEta = (*jt)->clusterRef()->positionREP().Eta();

      multitracks.linkedClusters.push_back(std::make_pair(clusterPhi, clusterEta));
    }

    it->first->setMultilinks(multitracks);
  }

  if (debug_) {
    std::cout << " Number of found links by tracks : " << std::endl;
    
    for (size_t i = 0; i < 10; ++i)
      std::cout << i << " : " << histo[i] << std::endl;
  }

  //debug_ = false;
}

void
KDTreeLinkerPSEcal::clear()
{
  psSet_.clear();
  ecalSet_.clear();

  rechitsNegSet_.clear();
  rechitsPosSet_.clear();

  rechitEcalLinks_.clear();
  PSEcalLinks_.clear();

  treeNeg_.clear();
  treePos_.clear();
}


// bool
// KDTreeLinkerPSEcal::isCorrectTrack(reco::PFBlockElement* track) const
// {
//   return tracksSet_.find(track) != tracksSet_.end();
// }

// bool
// KDTreeLinkerPSEcal::isEcalCluster(const reco::PFBlockElement* cluster) const
// {
//   return clustersSet_.find(cluster) != clustersSet_.end();
// }

// bool
// KDTreeLinkerPSEcal::isLinked(reco::PFBlockElement* track,
// 			  const reco::PFBlockElement* cluster) const
// {
//   BlockEltClusterMap::const_iterator ret = PSEcalLinks_.find(track);

//   if ((ret == PSEcalLinks_.end()) ||
//       (ret->second.find(cluster) == ret->second.end()))
//     return false;
//   return true;
// }

// void
// KDTreeLinkerPSEcal::printTrackLinks(reco::PFBlockElement* track)
// {
//   std::cout << " Track :" << *track << std::endl;

//   BlockEltClusterMap::iterator ret = PSEcalLinks_.find(track);

//   std::cout << "   Nbr of associated clusters = " << ret->second.size() << std::endl;

//   if (ret != PSEcalLinks_.end()) {
//     for (BlockEltSet_const::iterator it = ret->second.begin();
// 	 it != ret->second.end(); ++it)
//       std::cout << "     " << *(*it) << std::endl;
//   }  
// }
