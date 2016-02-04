#ifndef FMGLfromTKEfficiencyHandler_H
#define FMGLfromTKEfficiencyHandler_H

/** \file FastSimulation/ParamL3MuonProducer/interface/FMGLfromL3TKEfficiencyHandler.h
 * A FastSimulation class to kill FSimTrack's for efficiency modeling.
 * \author Andrea Perrotta
 * $Date: 2007/06/12 15:22:21 $
 */

class RandomEngine;
class SimTrack;

class FMGLfromTKEfficiencyHandler {

public:

  FMGLfromTKEfficiencyHandler(const RandomEngine * engine);
  ~FMGLfromTKEfficiencyHandler();

  bool kill(const SimTrack &);

private:
  const RandomEngine * random;  
  static const int nEtaBins=60;
  float Effic_Eta[nEtaBins];
};

#endif
