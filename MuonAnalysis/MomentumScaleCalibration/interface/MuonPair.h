#ifndef MuonPair_h
#define MuonPair_h

#include <TObject.h>
#include "DataFormats/Candidate/interface/Particle.h"

typedef reco::Particle::LorentzVector lorentzVector;

/**
 * Simple class used to save the muon pairs in a root tree. <br>
 * Includes the information on the run and event number.
 */

class MuonPair : public TObject
{
public:
  MuonPair() :
    mu1(lorentzVector(0,0,0,0)),
    mu2(lorentzVector(0,0,0,0)),
    run(0),
    event(0)
  {}

  MuonPair(const lorentzVector & inputMu1, const lorentzVector & inputMu2,
	   const unsigned int inputRun, const unsigned int inputEvent) :
    mu1(inputMu1),
    mu2(inputMu2),
    run(inputRun),
    event(inputEvent)
  {}

  /// Used to copy the content of another MuonPair
  void copy(const MuonPair & copyPair)
  {
    mu1 = copyPair.mu1;
    mu2 = copyPair.mu2;
    run = copyPair.run;
    event = copyPair.event;
  }
  
  lorentzVector mu1;
  lorentzVector mu2;
  UInt_t run;
  UInt_t event;
  
  ClassDef(MuonPair, 1)
};
ClassImp(MuonPair)

#endif
