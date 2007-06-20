#ifndef FML1EfficiencyHandler_H
#define FML1EfficiencyHandler_H

/** \class FML1EfficiencyHandler
 * Class to deal with L1 muon efficiency as a function of eta, phi and pt.
 *
  * \author Andrea Perrotta Date:  05/09/2006
 */

#include <cmath>

class RandomEngine;
class SimpleL1MuGMTCand;

class FML1EfficiencyHandler {

public:

  /// Constructor (read in the parametrizations from a data file) 
  FML1EfficiencyHandler(const RandomEngine * engine);
   
  /// Destructor
  ~FML1EfficiencyHandler();
   
  /// reject tracks according to parametrized algorithmic efficiency 
  bool kill(const SimpleL1MuGMTCand *);

private:

  
  const RandomEngine * random;
  static const int nEtaBins=120;
  static const int nPhiBins=100;
  double Effic_Eta[nEtaBins];
  double Effic_Phi_Barrel[nPhiBins];
  double Effic_Phi_Endcap[nPhiBins];
  double Effic_Phi_Extern[nPhiBins];


  inline double tuningfactor(int ieta) {
    double tf=0.;
    switch (ieta) {
    case 0: 
      tf = 1.045;
      break;
    case 1:
      tf = 1.049;
      break;
    case 2:  
      tf = 1.059;
      break;
    }
    return tf ;
  }


  inline double dumpingfactor(int ieta , float pt) {
    if (pt<3.) return 0.;
    //    float df=0.;
    double df=0.;
    switch (ieta) {
    case 0: 
      df = 1. - 1.260 * exp(-(pt-2.607)*(pt-2.607)) ;
      break;
    case 1:  
      df = 1. - 5.540 * exp(-(pt-1.401)*(pt-1.401)) ;
      break;
    case 2:  
      df = 1.;
      break;
    }
    return (df>0? df: 0.) ;
  }

};

#endif
