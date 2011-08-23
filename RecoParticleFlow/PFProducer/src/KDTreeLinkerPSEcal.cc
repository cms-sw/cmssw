#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerPSEcal.h"

#include "TMath.h"

using namespace KDTreeLinker;

KDTreeLinkerPSEcal::KDTreeLinkerPSEcal(double	phiOffset,
				       double ecalDiameter,
				       bool debug)
  : phiOffset_ (phiOffset), 
    ecalDiameter_(ecalDiameter),
    debug_(debug)
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
KDTreeLinkerPSEcal::insertTrack(reco::PFBlockElement* track)
{
  tracksSet_.insert(track);
}


void
KDTreeLinkerPSEcal::insertCluster(const reco::PFBlockElement* cluster,
				     const std::vector<reco::PFRecHitFraction> &fraction)
{
  // We create a list of cluster
  clustersSet_.insert(cluster);
  for(size_t rhit = 0; rhit < fraction.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fraction[rhit].recHitRef();
    double fract = fraction[rhit].fraction();

    if ((rh.isNull()) || (fract < 1E-4))
      continue;
      
    const reco::PFRecHit& rechit = *rh;
      
    // We save the links rechit to Clusters
    rhClustersLinks_[&rechit].insert(cluster);
    
    // We create a liste of rechits
    rechitsSet_.insert(&rechit);
  }
}

void 
KDTreeLinkerPSEcal::buildTree()
{
  // List of pseudo-rechits that will be used to create the KDTree
  std::vector<RHinfo> eltList;

  // Filling of this list
  for(RecHitSet::const_iterator it = rechitsSet_.begin(); 
      it != rechitsSet_.end(); it++) {
    
    const reco::PFRecHit::REPPoint &posrep = (*(*it)).positionREP();
    
    RHinfo rh1 (*it, posrep.Eta(), posrep.Phi());
    eltList.push_back(rh1);
    
    // Here we solve the problem of phi circular set by duplicating some rechits
    // too close to -Pi (or to Pi) and adding (substracting) to them 2 * Pi.
    if (rh1.phi > (M_PI - getPhiOffset())) {
      double phi = rh1.phi - 2 * M_PI;
      RHinfo rh2(*it, posrep.Eta(), phi); 
      eltList.push_back(rh2);
    }

    if (rh1.phi < (M_PI * -1.0 + getPhiOffset())) {
      double phi = rh1.phi + 2 * M_PI;
      RHinfo rh3(*it, posrep.Eta(), phi); 
      eltList.push_back(rh3);
    }
  }

  // Here we define the upper/lower bounds of the 2D space (eta/phi).
  double phimin = -1.0 * M_PI - getPhiOffset();
  double phimax = M_PI + getPhiOffset();
  TBox region(-3.0, 3.0, phimin, phimax);

  // We may now build the KDTree
  tree_.build(eltList, region);
}

void
KDTreeLinkerPSEcal::searchLinks()
{
  double resPSpitch = 0.19;
  double resPSlength = 6.1;

  // Must of the code has been taken from LinkByRecHit.cc


  // We iterate over the tracks.
  for(BlockEltSet::iterator it = tracksSet_.begin(); 
      it != tracksSet_.end(); it++) {
	
    reco::PFClusterRef clusterPSRef = (*it)->clusterRef();
    reco::PFCluster clusterPS = *clusterPSRef;
    (*it)->setIsValidMultilinks(true);

    // PS cluster position, extrapolated to ECAL
    double zPS = clusterPS.position().Z();
    double xPS = clusterPS.position().X(); //* zECAL/zPS;
    double yPS = clusterPS.position().Y(); //* zECAL/zPS;

    double deltaX = 0.;
    double deltaY = 0.;
    switch (clusterPS.layer()) {
    case PFLayer::PS1:
      // vertical strips, measure x with pitch precision
      deltaX = resPSpitch;
      deltaY = resPSlength;
      break;
    case PFLayer::PS2:
      // horizontal strips, measure y with pitch precision
      deltaY = resPSpitch;
      deltaX = resPSlength;
      break;
    default:
      break;
    }
 
    
    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    double rangeX = getEcalDiameter() * (0.05 + 1.0 / getEcalDiameter() * deltaX / 2.); 
    double rangeY = getEcalDiameter() * (0.05 + 1.0 / getEcalDiameter() * deltaY / 2.); 
    double range = std::max(rangeX, rangeY);
    
    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<RHinfo> recHits;
    double etaPS = clusterPS.positionREP().Eta();
    double phiPS = clusterPS.positionREP().Phi();

    TBox trackBox(etaPS-range, etaPS+range, 
		  phiPS-range, phiPS+range);
    tree_.search(trackBox, recHits);


    for(std::vector<RHinfo>::const_iterator rhit = recHits.begin(); 
	rhit != recHits.end(); ++rhit) {
           
      const std::vector< math::XYZPoint >& corners      = rhit->ptr->getCornersXYZ();
      if(corners.size() != 4) continue;

      // Find all clusters associated to given rechit
      RecHitClusterMap::iterator ret = rhClustersLinks_.find(rhit->ptr);
      
      for(BlockEltSet_const::const_iterator clusterIt = ret->second.begin(); 
	  clusterIt != ret->second.end(); clusterIt++) {
	
	reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
	double clusterz = clusterref->position().Z();

	const math::XYZPoint& posxyz = rhit->ptr->position() * zPS / clusterz;

	if (clusterref->layer() == PFLayer::ECAL_ENDCAP) {
	  if (clusterz * zPS < 0.) continue;

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
	    trackClusterLinks_[*it].insert(*clusterIt);


	} else
	  //// TODO YG : should never happend. Check?
	  continue;

	
      }
    }

  }
}

void
KDTreeLinkerPSEcal::updateTracksWithLinks()
{
  // For debug purpose
  std::vector<int> histo(25, 0);

  //TODO YG : Check if cluster positionREP() is valid

  // Here we save in each track the list of phi/eta values of linked clusters.
  for (BlockEltClusterMap::iterator it = trackClusterLinks_.begin();
       it != trackClusterLinks_.end(); ++it) {
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
}

void
KDTreeLinkerPSEcal::clear()
{
  tracksSet_.clear();
  rechitsSet_.clear();
  rhClustersLinks_.clear();
  clustersSet_.clear();
  trackClusterLinks_.clear();
  tree_.clear();
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
//   BlockEltClusterMap::const_iterator ret = trackClusterLinks_.find(track);

//   if ((ret == trackClusterLinks_.end()) ||
//       (ret->second.find(cluster) == ret->second.end()))
//     return false;
//   return true;
// }

// void
// KDTreeLinkerPSEcal::printTrackLinks(reco::PFBlockElement* track)
// {
//   std::cout << " Track :" << *track << std::endl;

//   BlockEltClusterMap::iterator ret = trackClusterLinks_.find(track);

//   std::cout << "   Nbr of associated clusters = " << ret->second.size() << std::endl;

//   if (ret != trackClusterLinks_.end()) {
//     for (BlockEltSet_const::iterator it = ret->second.begin();
// 	 it != ret->second.end(); ++it)
//       std::cout << "     " << *(*it) << std::endl;
//   }  
// }
