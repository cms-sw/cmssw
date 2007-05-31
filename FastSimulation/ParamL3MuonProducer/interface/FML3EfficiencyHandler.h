#ifndef FML3EfficiencyHandler_H
#define FML3EfficiencyHandler_H

/** \file FastSimulation/ParamL3MuonProducer/interface/FML3EfficiencyHandler.h
 * A FastSimulation class to kill FSimTrack's for efficiency modeling.
 * \author Andrea Perrotta   Date: 2007/05/22 13:57:19 $
 */

class RandomEngine; 
class FSimTrack;

class FML3EfficiencyHandler {

public:

  FML3EfficiencyHandler(const RandomEngine * engine);
  ~FML3EfficiencyHandler();

  bool kill(const FSimTrack &);

private:
 
  const RandomEngine * random;
  static const int nEtaBins=120;
  static const int nPhiBins=100;
  double Effic_Eta[nEtaBins];
  double Effic_Phi_Barrel[nPhiBins];
  double Effic_Phi_Endcap[nPhiBins];
  double Effic_Phi_Extern[nPhiBins];

};

#endif
