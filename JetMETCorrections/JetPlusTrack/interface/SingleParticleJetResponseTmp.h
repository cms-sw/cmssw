#ifndef SingleParticleJetResponseTmp_h
#define SingleParticleJetResponseTmp_h

/** \class SingleParticleJetResponse
    \brief This class computes the expected response in the calorimeters 
     for a given track, for which it is known if it has interacted, 
     its energy as measured in the tracker and the energy released 
     in the ECAL. Two algorithms are available at the moment

    @author O. Kodolova, A.Heister
 */
#include <fstream>
#include <sstream>
#include <ostream>
      
#include <vector>
#include <cmath>

class SingleParticleJetResponseTmp 
{
public:
  
  SingleParticleJetResponseTmp();
  ~SingleParticleJetResponseTmp(){};
  std::vector<double> response(double echar, double energycluster, int algo = 0) const;
};
#endif






