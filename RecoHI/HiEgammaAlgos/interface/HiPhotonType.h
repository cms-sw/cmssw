#ifndef HiPhotonType_h
#define HiPhotonType_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>

class HiGammaJetSignalDef
{
 public:

  HiGammaJetSignalDef();
  HiGammaJetSignalDef(const reco::GenParticleCollection *sigPartic);
  bool IsIsolated(const reco::GenParticle &pp)            ;
  bool IsIsolatedPP(const reco::GenParticle &pp)            ;
  bool IsIsolatedJP(const reco::GenParticle &pp)            ;

  //  bool IsSignal(const reco::Candidate &pp, double dPhi, bool isIso);
  //  int getIndex(const reco::Candidate &pp);
  double getDeltaR (const reco::Candidate &track1, const reco::Candidate &track2);
  double getDeltaPhi(const reco::Candidate &track1, const reco::Candidate &track2);
  double PI;

 private:
  const reco::GenParticleCollection        *fSigParticles;

};

class HiPhotonType
{
 public:
  HiPhotonType(edm::Handle<reco::GenParticleCollection> inputHandle);
  bool IsPrompt(const reco::GenParticle &pp);
  bool IsIsolated(const reco::GenParticle &pp);
  //  bool IsIsolatedPP(const reco::GenParticle &pp);
  //  bool IsIsolatedJP(const reco::GenParticle &pp);
  double PI;
 
 private:
  HiGammaJetSignalDef mcisocut;
};

#endif

