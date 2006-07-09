#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"

HcalHitMaker::HcalHitMaker(EcalHitMaker& grid,unsigned shower)
  :CaloHitMaker(grid.getCalorimeter(),DetId::Hcal,HcalHitMaker::getSubHcalDet(grid.getFSimTrack()),
		grid.getFSimTrack()->onHcal()?grid.getFSimTrack()->onHcal():grid.getFSimTrack()->onVFcal()+1,shower),
   myGrid(grid),  myTrack((grid.getFSimTrack()))
{
  //  std::cout << " Created HcalHitMaker " << std::endl;
  // normalize the direction
  ecalEntrance=myGrid.ecalEntrance();
  particleDirection=myTrack->ecalEntrance().vect().unit();
  radiusFactor=(EMSHOWER)? moliereRadius:interactionLength;
  mapCalculated=false;
  //std::cout << " Famos HCAL " << grid.getTrack()->onHcal() << " " <<  grid.getTrack()->onVFcal() << " " << showerType << std::endl;
  if(EMSHOWER&&(abs(grid.getFSimTrack()->type())!=11 && grid.getFSimTrack()->type()!=22))
    {
      std::cout << " FamosHcalHitMaker : Strange. The following shower has EM type" << std::endl <<* grid.getFSimTrack() << std::endl;
    }
}

bool HcalHitMaker::addHit(double r,double phi,unsigned layer)
{
  return true;
}

bool HcalHitMaker::setDepth(double depth)
{
  currentDepth=depth;
  return true;
}
