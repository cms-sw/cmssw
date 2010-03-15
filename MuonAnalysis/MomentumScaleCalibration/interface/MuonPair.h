#include <TObject.h>

#include "DataFormats/Candidate/interface/Particle.h"

typedef reco::Particle::LorentzVector lorentzVector;

/**
 * Simple class used to save the muon pairs in a root tree.
 */

class MuonPair : public TObject
{
public:
  lorentzVector mu1;
  lorentzVector mu2;
  ClassDef(MuonPair, 1)
};
ClassImp(MuonPair)
