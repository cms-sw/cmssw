//FAMOS header
#include "FastSimulation/ShowerDevelopment/interface/RadialInterval.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include <cmath>

RadialInterval::RadialInterval(double RC,unsigned nSpots,double energy,
			       const RandomEngine* engine)
  :
  theR(RC),theNumberOfSpots(nSpots),theSpotEnergy(energy),random(engine)
{

  currentRad=0.;
  currentEnergyFraction=0.;
  currentUlim=0.;
  nInter=0;
}

void RadialInterval::addInterval(double radius, double spotf)
{
  double radiussq=radius*radius;
  double enFrac=energyFractionInRadius(radius);
  if(radius>10) enFrac=1.;
  double energyFrac=enFrac-currentEnergyFraction;
  currentEnergyFraction=enFrac;
  //  std::cout << " Rad " << nInter << " Energy frac " << energyFrac << std::endl;
  
  // Calculates the number of spots. Add binomial fluctuations
  double dspot = random->gaussShoot(theNumberOfSpots*energyFrac,
			    sqrt(energyFrac*(1.-energyFrac)*theNumberOfSpots));
  //  std::cout << " Normal : " << theNumberOfSpots*energyFrac << " "<< dspot << std::endl;
  unsigned nspot=(unsigned)(dspot*spotf+0.5);
//  if(nspot<100) 
//    {
//      spotf=1.;
//      nspot=(unsigned)(theNumberOfSpots*energyFrac+0.5);
//    }
  
  dspotsunscaled.push_back(dspot);
  spotfraction.push_back(spotf);

  double spotEnergy=theSpotEnergy/spotf;
  //  std::cout << " The number of spots " << theNumberOfSpots << " " << nspot << std::endl;

  // This is not correct for the last interval, but will be overriden later
  nspots.push_back(nspot);
  spotE.push_back(spotEnergy);
  // computes the limits
  double umax = radiussq/(radiussq+theR*theR);
  if(radius>10)
    {
      umax=1.;
    }

  // Stores the limits
  uMax.push_back(umax);		 
  uMin.push_back(currentUlim);		 
  currentUlim=umax;

  // Stores the energy
  //  std::cout << " SpotE " << theSpotEnergy<< " " << spotf << " " << theSpotEnergy/spotf<< std::endl;

  ++nInter;
}

double RadialInterval::energyFractionInRadius( double rm)
{
  double rm2=rm*rm;
  return (rm2/(rm2+theR*theR));
}

void RadialInterval::compute()
{
  //  std::cout << " The number of Spots " << theNumberOfSpots << std::endl;
  //  std::cout << " Size : " << nInter << " " << nspots.size() << " " << dspotsunscaled.size() << std::endl;
  double ntotspots=0.;
  for(unsigned irad=0;irad<nInter-1;++irad)
    {
      ntotspots+=dspotsunscaled[irad];
      //    std::cout << " In the loop " << ntotspots << std::endl;
    }

  // This can happen (fluctuations)
  if(ntotspots>theNumberOfSpots) ntotspots=(double)theNumberOfSpots;
  //  std::cout << " Sous-total " << ntotspots << std::endl;
  dspotsunscaled[nInter-1]=(double)theNumberOfSpots-ntotspots;
  
  nspots[nInter-1]=(unsigned)(dspotsunscaled[nInter-1]*spotfraction[nInter-1]+0.5);
  //    std::cout << " Nlast " << nspots[nInter-1] << std::endl;
}
