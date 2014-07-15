#include "KDTreeLinkerTrackHcal.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TMath.h"

// the text name is different so that we can easily
// construct it when calling the factory
DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeLinkerTrackHcal, 
		  "KDTreeTrackAndHCALLinker"); 


KDTreeLinkerTrackHcal::KDTreeLinkerTrackHcal()
  : KDTreeLinkerBase()
{
  setCristalPhiEtaMaxSize(0.2);
  setPhiOffset(0.32);
}

KDTreeLinkerTrackHcal::~KDTreeLinkerTrackHcal()
{
  clear();
}

void
KDTreeLinkerTrackHcal::insertTargetElt(reco::PFBlockElement	*track)
{
  if( track->trackRefPF()->extrapolatedPoint( reco::PFTrajectoryPoint::HCALEntrance ).isValid() ) {
    targetSet_.insert(track);
  }
}


void
KDTreeLinkerTrackHcal::insertFieldClusterElt(reco::PFBlockElement	*hcalCluster)
{
  reco::PFClusterRef clusterref = hcalCluster->clusterRef();

  // This test is more or less done in PFBlockAlgo.h. In others cases, it should be switch on.
  //   if (!((clusterref->layer() == PFLayer::HCAL_ENDCAP) ||
  // 	(clusterref->layer() == PFLayer::HCAL_BARREL1)))
  //     return;

  const std::vector<reco::PFRecHitFraction> &fraction = clusterref->recHitFractions();

  // We create a list of hcalCluster
  fieldClusterSet_.insert(hcalCluster);
  for(size_t rhit = 0; rhit < fraction.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fraction[rhit].recHitRef();
    double fract = fraction[rhit].fraction();

    if ((rh.isNull()) || (fract < 1E-4))
      continue;
      
    const reco::PFRecHit& rechit = *rh;
      
    // We save the links rechit to HcalClusters
    rechit2ClusterLinks_[&rechit].insert(hcalCluster);
    
    // We create a liste of rechits
    rechitsSet_.insert(&rechit);
  }
}

void 
KDTreeLinkerTrackHcal::buildTree()
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
    if (rh1.dim2 > (M_PI - getPhiOffset())) {
      double phi = rh1.dim2 - 2 * M_PI;
      KDTreeNodeInfo rh2(*it, posrep.Eta(), phi); 
      eltList.push_back(rh2);
    }

    if (rh1.dim2 < (M_PI * -1.0 + getPhiOffset())) {
      double phi = rh1.dim2 + 2 * M_PI;
      KDTreeNodeInfo rh3(*it, posrep.Eta(), phi); 
      eltList.push_back(rh3);
    }
  }

  // Here we define the upper/lower bounds of the 2D space (eta/phi).
  double phimin = -1.0 * M_PI - getPhiOffset();
  double phimax = M_PI + getPhiOffset();

  // etamin-etamax, phimin-phimax
  KDTreeBox region(-3.0, 3.0, phimin, phimax);

  // We may now build the KDTree
  tree_.build(eltList, region);
}

void
KDTreeLinkerTrackHcal::searchLinks()
{
  // Must of the code has been taken from LinkByRecHit.cc

  // We iterate over the tracks.
  for(BlockEltSet::iterator it = targetSet_.begin(); 
      it != targetSet_.end(); it++) {
	
    reco::PFRecTrackRef trackref = (*it)->trackRefPF();

    const reco::PFTrajectoryPoint& atHCAL = 
      trackref->extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
    const reco::PFTrajectoryPoint& atHCALExit = 
      trackref->extrapolatedPoint(reco::PFTrajectoryPoint::HCALExit);
    
    // The track didn't reach hcal
    if( ! atHCAL.isValid()) continue;
    
    double dHEta = atHCALExit.positionREP().Eta() - atHCAL.positionREP().Eta();
    double dHPhi = atHCALExit.positionREP().Phi() - atHCAL.positionREP().Phi(); 
    if ( dHPhi > M_PI ) dHPhi = dHPhi - 2. * M_PI;
    else if ( dHPhi < -M_PI ) dHPhi = dHPhi + 2. * M_PI; 
    
    double tracketa = atHCAL.positionREP().Eta() + 0.1 * dHEta;
    double trackphi = atHCAL.positionREP().Phi() + 0.1 * dHPhi;
    
    if (trackphi > M_PI) trackphi -= 2 * M_PI;
    else if (trackphi < -M_PI) trackphi += 2 * M_PI;

    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    double inflation = 1.;
    double rangeEta = (getCristalPhiEtaMaxSize() * (1.5 + 0.5) + 0.2 * fabs(dHEta)) * inflation; 
    double rangePhi = (getCristalPhiEtaMaxSize() * (1.5 + 0.5) + 0.2 * fabs(dHPhi)) * inflation; 

    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<KDTreeNodeInfo> recHits;
    KDTreeBox trackBox(tracketa - rangeEta, tracketa + rangeEta, 
		       trackphi - rangePhi, trackphi + rangePhi);
    tree_.search(trackBox, recHits);
    
    // Here we check all rechit candidates using the non-approximated method.
    for(std::vector<KDTreeNodeInfo>::const_iterator rhit = recHits.begin(); 
	rhit != recHits.end(); ++rhit) {

      const reco::PFRecHit::REPPoint &rhrep		   = rhit->ptr->positionREP();
      const std::vector<reco::PFRecHit::REPPoint>& corners = rhit->ptr->getCornersREP();
      if(corners.size() != 4) continue;
      
      double rhsizeEta = fabs(corners[0].Eta() - corners[2].Eta());
      double rhsizePhi = fabs(corners[0].Phi() - corners[2].Phi());
      if ( rhsizePhi > M_PI ) rhsizePhi = 2.*M_PI - rhsizePhi;
      
      double deta = fabs(rhrep.Eta() - tracketa);
      double dphi = fabs(rhrep.Phi() - trackphi);
      if ( dphi > M_PI ) dphi = 2.*M_PI - dphi;
      
      // Find all clusters associated to given rechit
      RecHit2BlockEltMap::iterator ret = rechit2ClusterLinks_.find(rhit->ptr);
      
      for(BlockEltSet::iterator clusterIt = ret->second.begin(); 
	  clusterIt != ret->second.end(); clusterIt++) {
	
	const reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
	int fracsNbr = clusterref->recHitFractions().size();
	
	double _rhsizeEta = rhsizeEta * (1.5 + 0.5 / fracsNbr) + 0.2 * fabs(dHEta);
	double _rhsizePhi = rhsizePhi * (1.5 + 0.5 / fracsNbr) + 0.2 * fabs(dHPhi);
	
	// Check if the track and the cluster are linked
	if(deta < (_rhsizeEta / 2.) && dphi < (_rhsizePhi / 2.))
	  cluster2TargetLinks_[*clusterIt].insert(*it);
      }
    }
  }
}

void
KDTreeLinkerTrackHcal::updatePFBlockEltWithLinks()
{
  //TODO YG : Check if cluster positionREP() is valid ?

  // Here we save in each HCAL cluster the list of phi/eta values of linked clusters.
  for (BlockElt2BlockEltMap::iterator it = cluster2TargetLinks_.begin();
       it != cluster2TargetLinks_.end(); ++it) {
    reco::PFMultiLinksTC multitracks(true);

    for (BlockEltSet::iterator jt = it->second.begin();
	 jt != it->second.end(); ++jt) {

      reco::PFRecTrackRef trackref = (*jt)->trackRefPF();
      const reco::PFTrajectoryPoint& atHCAL = 
	trackref->extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
      double tracketa = atHCAL.positionREP().Eta();
      double trackphi = atHCAL.positionREP().Phi();
      
      multitracks.linkedClusters.push_back(std::make_pair(trackphi, tracketa));
    }

    it->first->setMultilinks(multitracks);
  }

  // We set the multilinks flag of the track to true. It will allow us to 
  // use in an optimized way our algo results in the recursive linking algo.
  for (BlockEltSet::iterator it = fieldClusterSet_.begin();
       it != fieldClusterSet_.end(); ++it)
    (*it)->setIsValidMultilinks(true);

}

void
KDTreeLinkerTrackHcal::clear()
{
  targetSet_.clear();
  fieldClusterSet_.clear();

  rechitsSet_.clear();

  rechit2ClusterLinks_.clear();
  cluster2TargetLinks_.clear();

  tree_.clear();
}
