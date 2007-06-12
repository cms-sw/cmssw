#ifndef FMGLfromL3TKEfficiencyHandler_H
#define FMGLfromL3TKEfficiencyHandler_H

/** \file FastSimulation/ParamL3MuonProducer/interface/FMGLfromL3TKEfficiencyHandler.h
 * A FastSimulation class to kill FSimTrack's for efficiency modeling.
 * \author Andrea Perrotta
 * $Date: 2007/05/31 13:34:06 $
 */

class RandomEngine;
class SimTrack;

class FMGLfromL3TKEfficiencyHandler {

public:

  FMGLfromL3TKEfficiencyHandler(const RandomEngine * engine);
  ~FMGLfromL3TKEfficiencyHandler();

  bool kill(const SimTrack &);

private:
  const RandomEngine * random;
  static const int nEtaBins=120;
  float Effic_Eta[nEtaBins];
};

#endif
