/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef ParticleMassHelper_h
#define ParticleMassHelper_h

// system include files                                                                                                                                                                                              
#include <memory>

class ParticleMassHelper {

public:
  // constructor and Destructor
  ParticleMassHelper();
  virtual ~ParticleMassHelper();

  inline double Get_piMass(){return piMass;}
  inline double Get_tauMass(){return tauMass;}
  inline double Get_nuMass(){return nuMass;}

private:
  static const double piMass;
  static const double tauMass;
  static const double nuMass;
};

#endif
