#include "RecoEcal/EgammaClusterAlgos/interface/PositionAwareHit.h"

PositionAwareHit::PositionAwareHit(EcalRecHit &the_rechit, const CaloSubdetectorGeometry *the_geometry)
{
  rechit = the_rechit;
  
  const CaloCellGeometry *this_cell = the_geometry->getGeometry(getId());
  position = this_cell->getPosition();
  
  used = false;
}

int PositionAwareHit::operator<(const PositionAwareHit &other_hit) const
{
  if(other_hit.getEnergy() > getEnergy()) 
    return 0;
  else
    return 1;
}

