#include "KDTreeLinkerTrackEcal.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TMath.h"

// the text name is different so that we can easily
// construct it when calling the factory
DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeLinkerTrackEcal, 
		  "KDTreeTrackAndECALLinker"); 


KDTreeLinkerTrackEcal::KDTreeLinkerTrackEcal()
  : KDTreeLinkerBase()
{}

KDTreeLinkerTrackEcal::~KDTreeLinkerTrackEcal()
{
  clear();
}

void
KDTreeLinkerTrackEcal::insertTargetElt(reco::PFBlockElement	*track)
{
  if( track->trackRefPF()->extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() ) {
    targetSet_.insert(track);
  }
}


void
KDTreeLinkerTrackEcal::insertFieldClusterElt(reco::PFBlockElement	*ecalCluster)
{
  reco::PFClusterRef clusterref = ecalCluster->clusterRef();

  // This test is more or less done in PFBlockAlgo.h. In others cases, it should be switch on.
  //   if (!((clusterref->layer() == PFLayer::ECAL_ENDCAP) ||
  // 	(clusterref->layer() == PFLayer::ECAL_BARREL)))
  //     return;

  const std::vector<reco::PFRecHitFraction> &fraction = clusterref->recHitFractions();

  // We create a list of ecalCluster
  fieldClusterSet_.insert(ecalCluster);
  for(size_t rhit = 0; rhit < fraction.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fraction[rhit].recHitRef();
    double fract = fraction[rhit].fraction();

    if ((rh.isNull()) || (fract < 1E-4))
      continue;
      
    const reco::PFRecHit& rechit = *rh;
      
    // We save the links rechit to EcalClusters
    rechit2ClusterLinks_[&rechit].insert(ecalCluster);
    
    // We create a liste of rechits
    rechitsSet_.insert(&rechit);
  }
}

void 
KDTreeLinkerTrackEcal::buildTree()
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
KDTreeLinkerTrackEcal::searchLinks()
{
  // Must of the code has been taken from LinkByRecHit.cc

  // We iterate over the tracks.
  for(BlockEltSet::iterator it = targetSet_.begin(); 
      it != targetSet_.end(); it++) {
	
    reco::PFRecTrackRef trackref = (*it)->trackRefPF();

    // We set the multilinks flag of the track to true. It will allow us to 
    // use in an optimized way our algo results in the recursive linking algo.
    (*it)->setIsValidMultilinks(true);

    const reco::PFTrajectoryPoint& atECAL = 
      trackref->extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax);

    // The track didn't reach ecal
    if( ! atECAL.isValid() ) continue;
    
    const reco::PFTrajectoryPoint& atVertex = 
      trackref->extrapolatedPoint( reco::PFTrajectoryPoint::ClosestApproach );
    
    double trackPt = sqrt(atVertex.momentum().Vect().Perp2());
    double tracketa = atECAL.positionREP().Eta();
    double trackphi = atECAL.positionREP().Phi();
    double trackx = atECAL.position().X();
    double tracky = atECAL.position().Y();
    double trackz = atECAL.position().Z();
    
    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    double range = getCristalPhiEtaMaxSize() * (2.0 + 1.0 / std::min(1., trackPt / 2.)); 

    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<KDTreeNodeInfo> recHits;
    KDTreeBox trackBox(tracketa-range, tracketa+range, trackphi-range, trackphi+range);
    tree_.search(trackBox, recHits);
    
    // Here we check all rechit candidates using the non-approximated method.
    for(std::vector<KDTreeNodeInfo>::const_iterator rhit = recHits.begin(); 
	rhit != recHits.end(); ++rhit) {
           
      const std::vector< math::XYZPoint >& cornersxyz      = rhit->ptr->getCornersXYZ();
      const math::XYZPoint& posxyz			   = rhit->ptr->position();
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
      
      for(BlockEltSet::const_iterator clusterIt = ret->second.begin(); 
	  clusterIt != ret->second.end(); clusterIt++) {
	
	reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
	double clusterz = clusterref->position().Z();
	int fracsNbr = clusterref->recHitFractions().size();

	if (clusterref->layer() == PFLayer::ECAL_BARREL){ // BARREL
	  // Check if the track is in the barrel
	  if (fabs(trackz) > 300.) continue;

	  double _rhsizeEta = rhsizeEta * (2.00 + 1.0 / (fracsNbr * std::min(1.,trackPt/2.)));
	  double _rhsizePhi = rhsizePhi * (2.00 + 1.0 / (fracsNbr * std::min(1.,trackPt/2.)));
	  
	  // Check if the track and the cluster are linked
	  if(deta < (_rhsizeEta / 2.) && dphi < (_rhsizePhi / 2.))
	    target2ClusterLinks_[*it].insert(*clusterIt);

	  
	} else { // ENDCAP

	  // Check if the track is in the cap
	  if (fabs(trackz) < 300.) continue;
	  if (trackz * clusterz < 0.) continue;
	  
	  double x[5];
	  double y[5];
	  for ( unsigned jc=0; jc<4; ++jc ) {
	    math::XYZPoint cornerposxyz = cornersxyz[jc];
	    x[jc] = cornerposxyz.X() + (cornerposxyz.X()-posxyz.X())
	      * (1.00+0.50/fracsNbr /std::min(1.,trackPt/2.));
	    y[jc] = cornerposxyz.Y() + (cornerposxyz.Y()-posxyz.Y())
	      * (1.00+0.50/fracsNbr /std::min(1.,trackPt/2.));
	  }
	  
	  x[4] = x[0];
	  y[4] = y[0];
	  
	  bool isinside = TMath::IsInside(trackx,
					  tracky,
					  5,x,y);
	  
	  // Check if the track and the cluster are linked
	  if( isinside )
	    target2ClusterLinks_[*it].insert(*clusterIt);
	}
      }
    }
  }
}

void
KDTreeLinkerTrackEcal::updatePFBlockEltWithLinks()
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

void
KDTreeLinkerTrackEcal::clear()
{
  targetSet_.clear();
  fieldClusterSet_.clear();

  rechitsSet_.clear();

  rechit2ClusterLinks_.clear();
  target2ClusterLinks_.clear();

  tree_.clear();
}
