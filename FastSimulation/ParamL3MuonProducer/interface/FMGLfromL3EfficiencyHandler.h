#ifndef FMGLfromL3EfficiencyHandler_H
#define FMGLfromL3EfficiencyHandler_H

/** \file FastSimulation/ParamL3MuonProducer/interface/FML3EfficiencyHandler.h
 * A FastSimulation class to kill FSimTrack's for efficiency modeling.
 * \author Andrea Perrotta
 * $Date: 2007/05/22 13:57:19 $
 */

class RandomEngine;
class FSimTrack;

class FMGLfromL3EfficiencyHandler {

public:

  FMGLfromL3EfficiencyHandler(const RandomEngine * engine);
  ~FMGLfromL3EfficiencyHandler();

  bool kill(const FSimTrack &);

private:
  const RandomEngine * random;
  static const int nEtaBins=60;
  float Effic_Eta[nEtaBins];
};

#endif
