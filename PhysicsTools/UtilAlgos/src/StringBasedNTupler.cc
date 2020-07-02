#include "PhysicsTools/UtilAlgos/interface/StringBasedNTupler.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Hemisphere.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/METReco/interface/MET.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/METReco/interface/HcalNoiseSummary.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"

#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include <DataFormats/CaloRecHit/interface/CaloCluster.h>

#include <DataFormats/PatCandidates/interface/TriggerPath.h>

#include <DataFormats/PatCandidates/interface/PFParticle.h>
#include <SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h>


        #include <memory>

        
//--------------------------------------------------------------------------------
//just define here a list of objects you would like to be able to have a branch of
//--------------------------------------------------------------------------------
#define ANOTHER_VECTOR_CLASS(C) \
  if (class_ == #C)             \
  return StringBranchHelper<C>(*this, iEvent)()
#define ANOTHER_CLASS(C) \
  if (class_ == #C)      \
  return StringLeaveHelper<C>(*this, iEvent)()

TreeBranch::value TreeBranch::branch(const edm::Event& iEvent) {
  ANOTHER_VECTOR_CLASS(pat::Jet);
  else ANOTHER_VECTOR_CLASS(pat::Muon);
  else ANOTHER_VECTOR_CLASS(reco::GenParticle);
  else ANOTHER_VECTOR_CLASS(pat::Electron);
  else ANOTHER_VECTOR_CLASS(pat::MET);
  else ANOTHER_VECTOR_CLASS(pat::Tau);
  else ANOTHER_VECTOR_CLASS(pat::Hemisphere);
  else ANOTHER_VECTOR_CLASS(pat::Photon);
  else ANOTHER_VECTOR_CLASS(reco::Muon);
  else ANOTHER_VECTOR_CLASS(reco::Track);
  else ANOTHER_VECTOR_CLASS(reco::GsfElectron);
  else ANOTHER_VECTOR_CLASS(SimTrack);
  else ANOTHER_VECTOR_CLASS(l1extra::L1ParticleMap);
  else ANOTHER_VECTOR_CLASS(reco::Vertex);
  else ANOTHER_VECTOR_CLASS(pat::GenericParticle);
  else ANOTHER_VECTOR_CLASS(reco::MET);
  else ANOTHER_CLASS(edm::HepMCProduct);
  else ANOTHER_CLASS(reco::BeamSpot);
  else ANOTHER_CLASS(HcalNoiseSummary);
  else ANOTHER_CLASS(GenEventInfoProduct);
  else ANOTHER_VECTOR_CLASS(reco::HcalNoiseRBX);
  else ANOTHER_VECTOR_CLASS(reco::BasicJet);
  else ANOTHER_VECTOR_CLASS(reco::CaloJet);
  else ANOTHER_VECTOR_CLASS(reco::GenJet);
  else ANOTHER_VECTOR_CLASS(pat::TriggerPath);
  else ANOTHER_VECTOR_CLASS(reco::PFCandidate);
  else ANOTHER_VECTOR_CLASS(reco::CaloCluster);
  else {
    edm::LogError("TreeBranch") << branchName() << " failed to recognized class type: " << class_;
    return std::make_unique<std::vector<float>>();
  }
}
#undef ANOTHER_CLASS
