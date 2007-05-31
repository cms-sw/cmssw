#ifndef FMGLfromL3TKEfficiencyHandler_H
#define FMGLfromL3TKEfficiencyHandler_H

/** \file FastSimulation/ParamL3MuonProducer/interface/FMGLfromL3TKEfficiencyHandler.h
 * A FastSimulation class to kill FSimTrack's for efficiency modeling.
 * \author Andrea Perrotta
 * $Date: 2007/05/22 13:57:19 $
 */

class RandomEngine;
class FSimTrack;

class FMGLfromL3TKEfficiencyHandler {

public:

  FMGLfromL3TKEfficiencyHandler(const RandomEngine * engine);
  ~FMGLfromL3TKEfficiencyHandler();

  bool kill(const FSimTrack &);

private:
  const RandomEngine * random;
  static const int nEtaBins=120;
  float Effic_Eta[nEtaBins];
};

#endif
