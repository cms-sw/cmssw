#ifndef RecoTauTag_RecoTau_PositionAtECalEntrance_h
#define RecoTauTag_RecoTau_PositionAtECalEntrance_h

/** \class PositionAtECalEntrance
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

class PositionAtECalEntrance 
{
 public:
  PositionAtECalEntrance();
  ~PositionAtECalEntrance();

  void beginEvent(const edm::EventSetup&);

  reco::Candidate::Point operator()(const reco::Candidate* particle, bool& success) const;

 private:
  double bField_z_;
};

#endif // RecoTauTag_RecoTau_PositionAtECalEntrance_h
