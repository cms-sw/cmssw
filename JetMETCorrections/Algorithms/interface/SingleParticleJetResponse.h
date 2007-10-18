#ifndef SingleParticleJetResponse_h
#define SingleParticleJetResponse_h

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

class SingleParticleJetResponse 
{
public:
  
  SingleParticleJetResponse();
  ~SingleParticleJetResponse(){};
  std::vector<double> response(double echar, double energycluster, int algo = 0) const;
};
#endif






