#ifndef FMGLfromTKEfficiencyHandler_H
#define FMGLfromTKEfficiencyHandler_H

/** \file FastSimulation/ParamL3MuonProducer/interface/FMGLfromL3TKEfficiencyHandler.h
 * A FastSimulation class to kill FSimTrack's for efficiency modeling.
 * \author Andrea Perrotta
 * $Date: 2007/05/22 13:57:19 $
 */

class RandomEngine;
class FSimTrack;

class FMGLfromTKEfficiencyHandler {

public:

  FMGLfromTKEfficiencyHandler(const RandomEngine * engine);
  ~FMGLfromTKEfficiencyHandler();

  bool kill(const FSimTrack &);

private:
  const RandomEngine * random;  
  static const int nEtaBins=60;
  float Effic_Eta[nEtaBins];
};

#endif
