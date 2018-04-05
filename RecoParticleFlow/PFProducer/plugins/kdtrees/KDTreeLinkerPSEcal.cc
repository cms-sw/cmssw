#include "KDTreeLinkerPSEcal.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TMath.h"

// the text name is different so that we can easily
// construct it when calling the factory
DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeLinkerPSEcal, 
		  "KDTreePreshowerAndECALLinker"); 

KDTreeLinkerPSEcal::KDTreeLinkerPSEcal()
  : KDTreeLinkerBase(),
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
KDTreeLinkerPSEcal::insertTargetElt(reco::PFBlockElement	*psCluster)
{
  // This test is more or less done in PFBlockAlgo.h. In others cases, it should be switch on.
  //if (!((psCluster->clusterRef()->layer() == PFLayer::PS1) || (psCluster->clusterRef()->layer() == PFLayer::PS2)))
  //  return;
  targetSet_.insert(psCluster);
}


void
KDTreeLinkerPSEcal::insertFieldClusterElt(reco::PFBlockElement	*ecalCluster)
{
  const reco::PFClusterRef& clusterref = ecalCluster->clusterRef();

  if (clusterref->layer() != PFLayer::ECAL_ENDCAP)
    return;
	  
  const std::vector<reco::PFRecHitFraction> &fraction = clusterref->recHitFractions();

  // We create a list of cluster
  fieldClusterSet_.insert(ecalCluster);

  double clusterz = clusterref->position().Z();
  RecHitSet& rechitsSet = (clusterz < 0) ? rechitsNegSet_ : rechitsPosSet_;

  for(size_t rhit = 0; rhit < fraction.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fraction[rhit].recHitRef();
    double fract = fraction[rhit].fraction();

    if ((rh.isNull()) || (fract < 1E-4))
      continue;
      
    const reco::PFRecHit& rechit = *rh;
      
    // We save the links rechit to Clusters
    rechit2ClusterLinks_[&rechit].insert(ecalCluster);
    
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
KDTreeLinkerPSEcal::buildTree(const RecHitSet	&rechitsSet,
			      KDTreeLinkerAlgo	&tree)
{
  // List of pseudo-rechits that will be used to create the KDTree
  std::vector<KDTreeNodeInfo> eltList;

  // Filling of this eltList
  for(RecHitSet::const_iterator it = rechitsSet.begin(); 
      it != rechitsSet.end(); it++) {

    const reco::PFRecHit* rh = *it;
    const auto & posxyz = rh->position();
        
    KDTreeNodeInfo rhinfo (rh, posxyz.x(), posxyz.y());
    eltList.push_back(rhinfo);
  }

  // xmin-xmax, ymain-ymax
  KDTreeBox region(-150., 150., -150., 150.);

  // We may now build the KDTree
  tree.build(eltList, region);
}

void
KDTreeLinkerPSEcal::searchLinks()
{
  // Must of the code has been taken from LinkByRecHit.cc

  // We iterate over the PS clusters.
  for(BlockEltSet::iterator it = targetSet_.begin(); 
      it != targetSet_.end(); it++) {

    (*it)->setIsValidMultilinks(true);
	
    reco::PFClusterRef clusterPSRef = (*it)->clusterRef();
    const reco::PFCluster& clusterPS = *clusterPSRef;

    // PS cluster position, extrapolated to ECAL
    double zPS = clusterPS.position().Z();
    double xPS = clusterPS.position().X();
    double yPS = clusterPS.position().Y();

    double etaPS = fabs(clusterPS.positionREP().eta());
    double deltaX = 0.;
    double deltaY = 0.;
    double xPSonEcal = xPS;
    double yPSonEcal = yPS;

    if (clusterPS.layer() == PFLayer::PS1) { // PS1

      // vertical strips, measure x with pitch precision
      deltaX = resPSpitch_;
      deltaY = resPSlength_;
      xPSonEcal *= ps1ToEcal_;
      yPSonEcal *= ps1ToEcal_;

    } else { // PS2

      // horizontal strips, measure y with pitch precision
      deltaY = resPSpitch_;
      deltaX = resPSlength_;
      xPSonEcal *= ps2ToEcal_;
      yPSonEcal *= ps2ToEcal_;

    }
 
    
    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    
    
    double maxEcalRadius = getCristalXYMaxSize() / 2.;

    // The inflation factor includes the approximate projection from Preshower to ECAL
    double inflation = 2.4 - (etaPS-1.6);
    double rangeX = maxEcalRadius * (1 + (0.05 + 1.0 / maxEcalRadius * deltaX / 2.)) * inflation; 
    double rangeY = maxEcalRadius * (1 + (0.05 + 1.0 / maxEcalRadius * deltaY / 2.)) * inflation; 
    
    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<KDTreeNodeInfo> recHits;
    KDTreeBox trackBox(xPSonEcal - rangeX, xPSonEcal + rangeX, 
		  yPSonEcal - rangeY, yPSonEcal + rangeY);

    if (zPS < 0)
      treeNeg_.search(trackBox, recHits);
    else
      treePos_.search(trackBox, recHits);


    for(std::vector<KDTreeNodeInfo>::const_iterator rhit = recHits.begin(); 
	rhit != recHits.end(); ++rhit) {
           
      const auto & corners = rhit->ptr->getCornersXYZ();

      // Find all clusters associated to given rechit
      RecHit2BlockEltMap::iterator ret = rechit2ClusterLinks_.find(rhit->ptr);
      
      for(BlockEltSet::const_iterator clusterIt = ret->second.begin(); 
	  clusterIt != ret->second.end(); clusterIt++) {
	
	reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
	double clusterz = clusterref->position().z();

	const auto & posxyz = rhit->ptr->position() * zPS / clusterz;

	double x[5];
	double y[5];
	for ( unsigned jc=0; jc<4; ++jc ) {
	  auto cornerpos = corners[jc].basicVector() * zPS / clusterz;
	  x[3-jc] = cornerpos.x() + (cornerpos.x()-posxyz.x()) * (0.05 +1.0/fabs((cornerpos.x()-posxyz.x()))*deltaX/2.);
	  y[3-jc] = cornerpos.y() + (cornerpos.y()-posxyz.y()) * (0.05 +1.0/fabs((cornerpos.y()-posxyz.y()))*deltaY/2.);
	}

	x[4] = x[0];
	y[4] = y[0];
	
	bool isinside = TMath::IsInside(xPS,
					yPS,
					5,x,y);
	
	// Check if the track and the cluster are linked
	if( isinside )
	  target2ClusterLinks_[*it].insert(*clusterIt);	
      }
    }
    
  }
}

void
KDTreeLinkerPSEcal::updatePFBlockEltWithLinks()
{
  //TODO YG : Check if cluster positionREP() is valid ?

  // Here we save in each track the list of phi/eta values of linked clusters.
  for (BlockElt2BlockEltMap::iterator it = target2ClusterLinks_.begin();
       it != target2ClusterLinks_.end(); ++it) {
    reco::PFMultiLinksTC multitracks(true);

    for (BlockEltSet::iterator jt = it->second.begin();
	 jt != it->second.end(); ++jt) {

      double clusterphi = (*jt)->clusterRef()->positionREP().phi();
      double clustereta = (*jt)->clusterRef()->positionREP().eta();

      multitracks.linkedClusters.push_back(std::make_pair(clusterphi, clustereta));
    }

    it->first->setMultilinks(multitracks);
  }
}

void
KDTreeLinkerPSEcal::clear()
{
  targetSet_.clear();
  fieldClusterSet_.clear();

  rechitsNegSet_.clear();
  rechitsPosSet_.clear();

  rechit2ClusterLinks_.clear();
  target2ClusterLinks_.clear();

  treeNeg_.clear();
  treePos_.clear();
}
