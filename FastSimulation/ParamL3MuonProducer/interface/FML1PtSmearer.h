#ifndef FML1PtSmearer_H
#define FML1PtSmearer_H

#include <vector>
class SimpleL1MuGMTCand;
class RandomEngine; 

/** \class FML1PtSmearer
 * Class to deal with the 'smearing' of the L1 muon transverse momentum.
 * The output momentum is generated according to the probablility that  
 * a MC muon with the same pt leads to that (discrete) momentum value 
 * in the GMT. 
 *
 * \author Andrea Perrotta   Date: 26/11/2004
 *                                 05/09/2006
 */
class FML1PtSmearer {

public:

  /// Constructor (read in the parametrizations from a data file) 
  FML1PtSmearer(const RandomEngine * engine);
   
  /// Destructor
  ~FML1PtSmearer();
    
  /// smear the transverse momentum of a SimplL1MuGMTCand
  bool smear(SimpleL1MuGMTCand *);

private:

  const RandomEngine * random;

  int IndexOfPtgen(float pt);

  static const int NPTL1=31;
  static const int NPT=136;
  static const int DIMRES=3*NPT*NPTL1;
  float resolution[DIMRES];

  inline float ChargeMisIdent(int ieta , double pt) {
    float df=0.;
    switch (ieta) {
    case 0: 
      df =  2.16909e-03  + 1.95708e-04*pt ;
      break;
    case 1:  
      if (pt>500.) pt = 500.;
      df = 1.00445e-02 + 1.15253e-03*pt - 7.73819e-07*pt*pt ;
      break;
    case 2:
      if (pt>200.) pt = 200.;
      df = 5.00580e-02 + 5.88949e-03*pt - 2.98100e-05*pt*pt + 5.02454e-08*pt*pt*pt ;
      break;
    }
    return (df<0.5? df: 0.5) ;
  }
  
};

#endif
