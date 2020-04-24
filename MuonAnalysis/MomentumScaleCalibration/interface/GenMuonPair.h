#ifndef GenMuonPair_h
#define GenMuonPair_h

#include <TObject.h>
#include "DataFormats/Candidate/interface/Particle.h"
// #include "MuonAnalysis/MomentumScaleCalibration/interface/BaseMuonPair.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Muon.h"

typedef reco::Particle::LorentzVector lorentzVector;

/**
 * For simplicity we use a different class than for reco used to save the gen muon pairs in a root tree. <br>
 * The reason is that there is no need for a map in this case as there is only one collection of generated muons. <br>
 * Additionally we want to save the motherId (we could have done using one of the unsigned int in the MuonPair class,
 * but this way the name of the datamember is more explicative). <br>
 * If it will be needed, it will be straightforward to migrate also the genMuons to use the MuonPair class.
 */

class GenMuonPair : public TObject
{
public:
  GenMuonPair() :
    mu1(lorentzVector(0,0,0,0),-1),
    mu2(lorentzVector(0,0,0,0),1),
    motherId(0)//,
    //    statusMu(-1),
  {}

  GenMuonPair(const MuScleFitMuon & initMu1, const MuScleFitMuon & initMu2,
	      const int initMotherId) :
    //	      const int initMotherId, const int initStatusMu) :
    mu1(initMu1),
    mu2(initMu2),
    motherId(initMotherId)//,
    //    statusMu(initStatusMu)
    // ,
    // motherId(initMotherId)
  {
    // Put this in the initialization list and root will not compile...
    // Probably some conflict with the other MuonPair class that also contains integers or
    // something even weirder...
/*     motherId = initMotherId; */
  }

  /// Used to copy the content of another GenMuonPair
  void copy(const GenMuonPair & copyPair)
  {
    mu1 = copyPair.mu1;
    mu2 = copyPair.mu2;
    motherId = copyPair.motherId;
    //    statusMu = copyPair.statusMu;
  }

  MuScleFitMuon mu1;
  MuScleFitMuon mu2;
  Int_t motherId;
  //  Int_t statusMu;

  ClassDef(GenMuonPair, 3)
    };
ClassImp(GenMuonPair)
  
#endif
