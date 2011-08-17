#include "RecoParticleFlow/PFProducer/interface/KDTreeTrackEcalLinker.h"
#include "TMath.h"

using namespace KDTreeLinker;

TrackEcalLinker::TrackEcalLinker(double	phiOffset,
				 double ecalDiameter,
				 bool debug)
  : phiOffset_ (phiOffset), 
    ecalDiameter_(ecalDiameter),
    debug_(debug)
{}

TrackEcalLinker::~TrackEcalLinker()
{
  clear();
}

void
TrackEcalLinker::setPhiOffset(double phiOffset)
{
  phiOffset_ = phiOffset;
}

double
TrackEcalLinker::getPhiOffset() const
{
  return phiOffset_;
}

void
TrackEcalLinker::setEcalDiameter(double ecalDiameter)
{
  ecalDiameter_ = ecalDiameter;
}

double
TrackEcalLinker::getEcalDiameter() const
{
  return ecalDiameter_;
}

void 
TrackEcalLinker::setDebug(bool isDebug)
{
  debug_ = isDebug;
}

void
TrackEcalLinker::insertTrack(reco::PFBlockElement* track)
{
  tracksSet_.insert(track);
}


void
TrackEcalLinker::insertCluster(const reco::PFBlockElement* cluster,
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
TrackEcalLinker::buildTree()
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
TrackEcalLinker::searchLinks()
{
  // Must of the code has been taken from LinkByRecHit.cc

  // We iterate over the tracks.
  for(BlockEltSetNC::iterator it = tracksSet_.begin(); 
      it != tracksSet_.end(); it++) {
	
    reco::PFRecTrackRef trackref = (*it)->trackRefPF();
    reco::PFRecTrack track(*trackref);

    // We set the multilinks flag of the track to true. It will allow us to 
    // use in an optimized way our algo results in the recursive linking algo.
    (*it)->setIsValidMultilinks(true);

    // We fill the positionREP if necessary
    const reco::PFTrajectoryPoint& atECAL_tmp = 
      (*trackref).extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax );
    if(std::abs(atECAL_tmp.positionREP().Eta())<1E-9 &&
       std::abs(atECAL_tmp.positionREP().Phi())<1E-9 &&
       atECAL_tmp.positionREP().R()<1E-9) 
      track.calculatePositionREP();

    const reco::PFTrajectoryPoint& atECAL = 
      track.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax );

    // The track didn't reach ecal
    if( ! atECAL.isValid() ) continue;
    
    const reco::PFTrajectoryPoint& atVertex = 
      track.extrapolatedPoint( reco::PFTrajectoryPoint::ClosestApproach );

    double trackPt = sqrt(atVertex.momentum().Vect().Perp2());
    double tracketa = atECAL.positionREP().Eta();
    double trackphi = atECAL.positionREP().Phi();
    double trackx = atECAL.position().X();
    double tracky = atECAL.position().Y();
    double trackz = atECAL.position().Z();

    // Estimate the maximal envelope in phi/eta that will be used to find rechit candidates.
    // Same envelope for cap et barrel rechits.
    double range = getEcalDiameter() * (2.0 + 1.0 / std::min(1., trackPt / 2.)); 

    // We search for all candidate recHits, ie all recHits contained in the maximal size envelope.
    std::vector<RHinfo> recHits;
    TBox trackBox(tracketa-range, tracketa+range, trackphi-range, trackphi+range);
    tree_.search(trackBox, recHits);
    
    for(std::vector<RHinfo>::const_iterator rhit = recHits.begin(); 
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
      RecHitClusterMap::iterator ret = rhClustersLinks_.find(rhit->ptr);
      
      for(BlockEltSet::const_iterator clusterIt = ret->second.begin(); 
	  clusterIt != ret->second.end(); clusterIt++) {
	
	reco::PFClusterRef clusterref = (*clusterIt)->clusterRef();
	double clusterz = clusterref->position().Z();
	int fracsNbr = clusterref->recHitFractions().size();

	//BARREL
	if (clusterref->layer() == PFLayer::ECAL_BARREL){
	  // Check if the track is in the barrel
	  if (fabs(trackz) > 300.) continue;

	  double _rhsizeEta = rhsizeEta * (2.00 + 1.0 / (fracsNbr * std::min(1.,trackPt/2.)));
	  double _rhsizePhi = rhsizePhi * (2.00 + 1.0 / (fracsNbr * std::min(1.,trackPt/2.)));
	  
	  // Check if the track and the cluster are linked
	  if(deta < (_rhsizeEta / 2.) && dphi < (_rhsizePhi / 2.))
	    trackClusterLinks_[*it].insert(*clusterIt);

	  
	}
	//CAP 
	else if (clusterref->layer() == PFLayer::ECAL_ENDCAP) {
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
	    trackClusterLinks_[*it].insert(*clusterIt);


	} else
	  //// TODO YG : should never happend. Check?
	  continue;

	
      }
    }
  }
}

void
TrackEcalLinker::updateTracksWithLinks()
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

    for (BlockEltSet::iterator jt = it->second.begin();
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
TrackEcalLinker::clear()
{
  tracksSet_.clear();
  rechitsSet_.clear();
  rhClustersLinks_.clear();
  clustersSet_.clear();
  trackClusterLinks_.clear();
  tree_.clear();
}


bool
TrackEcalLinker::isCorrectTrack(reco::PFBlockElement* track) const
{
  return tracksSet_.find(track) != tracksSet_.end();
}

bool
TrackEcalLinker::isEcalCluster(const reco::PFBlockElement* cluster) const
{
  return clustersSet_.find(cluster) != clustersSet_.end();
}

bool
TrackEcalLinker::isLinked(reco::PFBlockElement* track,
			  const reco::PFBlockElement* cluster) const
{
  BlockEltClusterMap::const_iterator ret = trackClusterLinks_.find(track);

  if ((ret == trackClusterLinks_.end()) ||
      (ret->second.find(cluster) == ret->second.end()))
    return false;
  return true;
}

void
TrackEcalLinker::printTrackLinks(reco::PFBlockElement* track)
{
  std::cout << " Track :" << *track << std::endl;

  BlockEltClusterMap::iterator ret = trackClusterLinks_.find(track);

  std::cout << "   Nbr of associated clusters = " << ret->second.size() << std::endl;

  if (ret != trackClusterLinks_.end()) {
    for (BlockEltSet::iterator it = ret->second.begin();
	 it != ret->second.end(); ++it)
      std::cout << "     " << *(*it) << std::endl;
  }  
}
