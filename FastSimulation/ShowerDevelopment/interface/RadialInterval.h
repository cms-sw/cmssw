#ifndef RadialInterval_H
#define RadialInterval_H

//  Created 1/11/04. F. Beaudette (CERN)
//  This class is used to ease the lateral development with
// different spot fractions in FamosShower. 


#include <vector>

class RandomEngine;

class RadialInterval
{
 public:
  /// Standard constructor Rc: mean Radius
  RadialInterval(double RC,unsigned nSpots, double energy,
		 const RandomEngine* engine);
  ~RadialInterval(){;}
  
  /// Add an interval : first argument is the radius, the second is the
  /// fraction of spots in this interval R>10 <-> infinity
  void addInterval(double,double);
  /// Most of the calculation are made in addInterval
  /// but the normal number of spots has to be set 
  void compute();
  /// Number of intervals
  inline unsigned nIntervals() const { return nInter;}
  /// Spot energy in a given interval
  inline double getSpotEnergy(unsigned i) const { 
    //    std::cout << " getSpotEnergy " << i << " " << spotE.size() << std::endl;
    return spotE[i];}
  /// Number of spots in a given interval
  inline unsigned getNumberOfSpots(unsigned i) const { 
    //    std::cout << " getNumberOfSpots " << i << " " << nspots.size() << std::endl;
    return nspots[i];
  }
  /// Lower limit of the argument in the radius generator
  inline double getUmin(unsigned i) const {
    //    std::cout << " getUmin " << i << " " << uMin.size() << std::endl;
    return uMin[i];
  }
  /// Upper limit of the argument in the radius generator
  inline double getUmax(unsigned i) const {
    //    std::cout << " getUmax " << i << " " << uMax.size() << std::endl;
    return uMax[i];
  }

 private:
  double currentRad;
  double currentEnergyFraction;
  double currentUlim;
  double theR;
  unsigned theNumberOfSpots;
  double theSpotEnergy;
  unsigned nInter; 

  std::vector<double> uMin;
  std::vector<double> uMax;
  std::vector<unsigned> nspots;
  std::vector<double> spotE;
  std::vector<double> dspotsunscaled;
  std::vector<double> spotfraction;

 private:
    // Fraction of the energy in rm Moliere radius
  double energyFractionInRadius(double rm);

  // Famos Random Engine
  const RandomEngine* random;
  
};
#endif
