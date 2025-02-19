#include "RecoParticleFlow/PFClusterShapeProducer/interface/PFClusterShapeAlgo.h"

PFClusterShapeAlgo::PFClusterShapeAlgo(bool useFractions, double w0)
{
  useFractions_ = useFractions;
  w0_ = w0;
}

PFClusterShapeAlgo::~PFClusterShapeAlgo()
{
}

reco::ClusterShapeCollection * 
PFClusterShapeAlgo::makeClusterShapes(edm::Handle<reco::PFClusterCollection> clusterHandle,
				      edm::Handle<reco::PFRecHitCollection>   rechitHandle,
				      const CaloSubdetectorGeometry * the_barrelGeo_p,
				      const CaloSubdetectorTopology * the_barrelTop_p,
				      const CaloSubdetectorGeometry * the_endcapGeo_p,
				      const CaloSubdetectorTopology * the_endcapTop_p)
{
  static const float etaEndOfBarrel = 1.497;

  topoVector.push_back(the_barrelTop_p);
  topoVector.push_back(the_endcapTop_p);
  geomVector.push_back(the_barrelGeo_p);
  geomVector.push_back(the_endcapGeo_p);

  reco::ClusterShapeCollection * shape_v_p = new reco::ClusterShapeCollection();

  currentRecHit_v_p = rechitHandle;

  for (unsigned int i = 0; i < clusterHandle->size(); ++i)
    {
      // Make each cluster the "current" cluster
      currentCluster_p = reco::PFClusterRef(clusterHandle, i);
      currentClusterIndex_ = i;

      // Find the right topology to use with this cluster
      topoIndex = BARREL;
      geomIndex = BARREL;
      const math::XYZVector currentClusterPos(currentCluster_p->position());
      if (fabs(currentClusterPos.eta()) > etaEndOfBarrel)
	{
	  topoIndex = ENDCAP;
	  geomIndex = ENDCAP;
	}

      // Create the clustershape and push it into the vector
      shape_v_p->push_back(makeClusterShape());
    }

  topoVector.clear();
  topoVector.clear();
  geomVector.clear();
  geomVector.clear();

  return shape_v_p;
}

reco::ClusterShape PFClusterShapeAlgo::makeClusterShape()
{
  find_eMax_e2nd();
  fill5x5Map();
  
  find_e2x2();
  find_e3x2();
  find_e3x3();
  find_e4x4();
  find_e5x5();
  
  find_e2x5Right();
  find_e2x5Left();
  find_e2x5Top();
  find_e2x5Bottom();
  
  covariances();
  
  double dummyLAT = 0;
  double dummyEtaLAT = 0;
  double dummyPhiLAT = 0;
  double dummyA20 = 0;
  double dummyA42 = 0;

  std::vector<double> dummyEnergyBasketFractionEta_v;
  std::vector<double> dummyEnergyBasketFractionPhi_v;

  return reco::ClusterShape(covEtaEta_, covEtaPhi_, covPhiPhi_, 
			    eMax_, eMaxId_, e2nd_, e2ndId_,
			    e2x2_, e3x2_, e3x3_, e4x4_, e5x5_, 
			    e2x5Right_, e2x5Left_, e2x5Top_, e2x5Bottom_, e3x2Ratio_,
			    dummyLAT, dummyEtaLAT, dummyPhiLAT, dummyA20, dummyA42,
			    dummyEnergyBasketFractionEta_v, dummyEnergyBasketFractionPhi_v);
}


void PFClusterShapeAlgo::find_eMax_e2nd() 
{
  std::map<double, DetId> energyMap;

  // First get the RecHitFractions:
  const std::vector<reco::PFRecHitFraction> & fraction_v = currentCluster_p->recHitFractions();
  // For every one of them...
  for (std::vector<reco::PFRecHitFraction>::const_iterator it = fraction_v.begin(); it != fraction_v.end(); ++it)
    {
      // ...find the corresponding rechit:
      // const reco::PFRecHit & rechit = (*currentRecHit_v_p)[it->recHitIndex()];
      const reco::PFRecHitRef rechit = it->recHitRef();
      // ...and DetId:
      const DetId rechitDetId = DetId(rechit->detId());
      // Make the new Pair and put it in the map:
      energyMap[rechit->energy()] = rechitDetId;
    }
  // maps are sorted in ascending order so get the last two elements:
  std::map<double, DetId>::reverse_iterator it = energyMap.rbegin();
  eMax_   = it->first;
  eMaxId_ = it->second;
  it++;
  e2nd_   = it->first;
  e2ndId_ = it->second;
}

int PFClusterShapeAlgo::findPFRHIndexFromDetId(unsigned int id)
{
  int index = -1; // need some negative number
  for (unsigned int k = 0; k < currentRecHit_v_p->size(); ++k)
    {
      if ((*currentRecHit_v_p)[k].detId() == id)
	{
	  index = static_cast<int>(k);
	  break;
	}
    }
  return index;
}


const reco::PFRecHitFraction * PFClusterShapeAlgo::getFractionFromDetId(const DetId & id)
{
  const std::vector< reco::PFRecHitFraction > & fraction_v = currentCluster_p->recHitFractions();
  for (std::vector<reco::PFRecHitFraction>::const_iterator it = fraction_v.begin(); it != fraction_v.end(); ++it)
    {
      //const unsigned int rhIndex = it->recHitIndex();
      //reco::PFRecHitRef rh_p(currentRecHit_v_p, rhIndex);
      const reco::PFRecHitRef rh_p = it->recHitRef();
      const DetId rhDetId = DetId(rh_p->detId());
      if (rhDetId == id) 
	{ 
	  return &(*it); 
	}
    }
  return 0;
}


void PFClusterShapeAlgo::fill5x5Map()
{
  // first get a navigator to the central element
  CaloNavigator<DetId> position = CaloNavigator<DetId>(eMaxId_, topoVector[topoIndex]);

  meanPosition_ = math::XYZVector(0.0, 0.0, 0.0);
  totalE_ = 0;

  for (int i = 0; i < 5; ++i)
    {
      for (int j = 0; j < 5; ++j)
	{
	  position.home();
	  position.offsetBy(i - 2, j - 2);

	  RecHitWithFraction newEntry;
	  newEntry.detId = DetId(0);
	  newEntry.energy = 0.0;
	  newEntry.position = math::XYZVector(0, 0, 0);

	  if (*position != DetId(0)) // if this is a valid detId...
	    {
	      // ...find the corresponding PFRecHit index
	      const int index = findPFRHIndexFromDetId((*position).rawId());

	      if (index >= 0) // if a PFRecHit exists for this detId
		{
		  double fraction = 1.0;
		  if (useFractions_) // if the algorithm should use fractions
		    { 
		      fraction = 0.0;
		      const reco::PFRecHitFraction * fraction_p = getFractionFromDetId(*position);
		      if (fraction_p) { fraction = fraction_p->fraction(); }
		    }

		  const reco::PFRecHitRef rhRef(currentRecHit_v_p, index);
		  const math::XYZVector crystalPosition(rhRef->position());
		  const double energyFraction =  rhRef->energy() * fraction;

		  newEntry.detId = *position;
		  newEntry.energy = energyFraction;
		  newEntry.position = crystalPosition;

		  meanPosition_ = meanPosition_ + crystalPosition * energyFraction; 
		  totalE_ += energyFraction;
		}
	    }
	  map5x5[i][j] = newEntry;
	}
    }
  meanPosition_ /= totalE_;
}

double PFClusterShapeAlgo::addMapEnergies(int etaIndexLow, int etaIndexHigh, int phiIndexLow, int phiIndexHigh)
{
  const int etaLow  = etaIndexLow  + 2;
  const int etaHigh = etaIndexHigh + 2;
  const int phiLow  = phiIndexLow  + 2;
  const int phiHigh = phiIndexHigh + 2;

  double energy = 0;

  for (int i = etaLow; i <= etaHigh; ++i)
    {
      for (int j = phiLow; j <= phiHigh; ++j)
	{
	  energy += map5x5[i][j].energy;
	}
    }
  return energy;
}

void PFClusterShapeAlgo::find_e3x3()       { e3x3_       = addMapEnergies(-1, +1, -1, +1); }
void PFClusterShapeAlgo::find_e5x5()       { e5x5_       = addMapEnergies(-2, +2, -2, +2); }
void PFClusterShapeAlgo::find_e2x5Right()  { e2x5Right_  = addMapEnergies(-2, +2, +1, +2); }
void PFClusterShapeAlgo::find_e2x5Left()   { e2x5Left_   = addMapEnergies(-2, +2, -2, -1); }
void PFClusterShapeAlgo::find_e2x5Top()    { e2x5Top_    = addMapEnergies(-2, -1, -2, +2); }
void PFClusterShapeAlgo::find_e2x5Bottom() { e2x5Bottom_ = addMapEnergies(+1, +2, -2, +2); }

void PFClusterShapeAlgo::find_e4x4() 
{
  if (eMaxDir == SE) { e4x4_ = addMapEnergies(-2, +1, -2, +1); return; }
  if (eMaxDir == NE) { e4x4_ = addMapEnergies(-2, +1, -1, +2); return; }
  if (eMaxDir == SW) { e4x4_ = addMapEnergies(-1, +2, -2, +1); return; }
  if (eMaxDir == NW) { e4x4_ = addMapEnergies(-1, +2, -1, +2); return; }
}

void PFClusterShapeAlgo::find_e2x2() 
{
  std::map<double, Direction> directionMap;

  directionMap[addMapEnergies(-1, +0, -1, +0)] = SE;
  directionMap[addMapEnergies(-1, +0, +0, +1)] = NE;
  directionMap[addMapEnergies(+0, +1, -1, +0)] = SW;
  directionMap[addMapEnergies(+0, +1, +0, +1)] = NW;

  const std::map<double, Direction>::reverse_iterator eMaxDir_it = directionMap.rbegin();

  eMaxDir = eMaxDir_it->second;

  e2x2_ = eMaxDir_it->first;
}

void PFClusterShapeAlgo::find_e3x2() 
{
  // Find the direction of the highest energy neighbour
  std::map<double, Direction> directionMap;
  directionMap[map5x5[2][3].energy] = N; 
  directionMap[map5x5[2][1].energy] = S;
  directionMap[map5x5[1][2].energy] = E;
  directionMap[map5x5[3][2].energy] = W;
  // Maps are sorted in ascending order - get the last element
  const Direction dir = directionMap.rbegin()->second;

  if (dir == N) 
    {
      e3x2_ = addMapEnergies(-1, +1, +0, +1);
      const double numerator   = map5x5[3][2].energy + map5x5[1][2].energy;
      const double denominator = map5x5[1][3].energy + map5x5[3][3].energy + 0.5;
      e3x2Ratio_ = numerator / denominator;
    }
  else if (dir == S)
    {
      e3x2_ = addMapEnergies(-1, +1, -1, +0);
      const double numerator   = map5x5[3][2].energy + map5x5[1][2].energy;
      const double denominator = map5x5[1][1].energy + map5x5[3][1].energy + 0.5;
      e3x2Ratio_ = numerator / denominator;
    }
  else if (dir == W)
    {
      e3x2_ = addMapEnergies(+0, +1, -1, +1);
      const double numerator   = map5x5[2][3].energy + map5x5[2][1].energy;
      const double denominator = map5x5[3][3].energy + map5x5[3][1].energy + 0.5;
      e3x2Ratio_ = numerator / denominator;
    }
  else if (dir == E)
    {
      e3x2_ = addMapEnergies(-1, +0, -1, +1);
      const double numerator   = map5x5[2][3].energy + map5x5[2][1].energy;
      const double denominator = map5x5[1][1].energy + map5x5[1][3].energy + 0.5;
      e3x2Ratio_ = numerator / denominator;
    }
}

void PFClusterShapeAlgo::covariances()
{
  double numeratorEtaEta = 0;
  double numeratorEtaPhi = 0;
  double numeratorPhiPhi = 0;
  double denominator     = 0;

  for (int i = 0; i < 5; ++i)
    {
      for (int j = 0; j < 5; ++j)
	{
	  const math::XYZVector & crystalPosition(map5x5[i][j].position);
	  
	  double dPhi = crystalPosition.phi() - meanPosition_.phi();
	  if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
	  if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }

	  const double dEta = crystalPosition.eta() - meanPosition_.eta();
	  
	  const double w = std::max(0.0, w0_ + log(map5x5[i][j].energy / totalE_));
	  
	  denominator += w;
	  numeratorEtaEta += w * dEta * dEta;
	  numeratorEtaPhi += w * dEta * dPhi;
	  numeratorPhiPhi += w * dPhi * dPhi;
	}
    }

  covEtaEta_ = numeratorEtaEta / denominator;
  covEtaPhi_ = numeratorEtaPhi / denominator;
  covPhiPhi_ = numeratorPhiPhi / denominator;
}



