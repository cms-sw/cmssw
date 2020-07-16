#ifndef RecoTauTag_RecoTau_PositionAtECalEntranceComputer_h
#define RecoTauTag_RecoTau_PositionAtECalEntranceComputer_h

/** \class PositionAtECalEntranceComputer
 *
 * Extrapolate particle (charged or neutral) to ECAL entrance,
 * in order to compute the distance of the tau to ECAL cracks and/or dead ECAL channels
 *
 * \authors Fabio Colombo,
 *          Christian Veelken
 *
 *
 *
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class PositionAtECalEntranceComputer {
public:
  PositionAtECalEntranceComputer();
  ~PositionAtECalEntranceComputer();

  void beginEvent(const edm::EventSetup&);

  //To do: it seems to more practical to put this to the ES
  reco::Candidate::Point operator()(const reco::Candidate* particle, bool& success) const;

private:
  double bField_z_;
};

#endif  // RecoTauTag_RecoTau_PositionAtECalEntranceComputer_h
