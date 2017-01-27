#include "RecoParticleFlow/PFProducer/plugins/PFGSFixLinker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonEqual.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef edm::ValueMap<reco::PFCandidatePtr> PFPtrMap;

PFGSFixLinker::PFGSFixLinker(edm::ParameterSet const& _config) :
  electronsMapName_(_config.getParameter<std::string>("ValueMapElectrons")),
  photonsMapName_(_config.getParameter<std::string>("ValueMapPhotons"))
{
  getToken(inputCandidatesToken_, _config, "PFCandidate");
  getToken(inputElectronsToken_, _config, "GsfElectrons");
  getToken(inputPhotonsToken_, _config, "Photons");
  getToken(electronMapToken_, _config, "GsfElectrons");
  getToken(photonMapToken_, _config, "Photons");
  
  produces<reco::PFCandidateCollection>();
  // new to old
  produces<PFPtrMap>();

  // e/g collections to PF
  produces<PFPtrMap>(electronsMapName_);
  produces<PFPtrMap>(photonsMapName_);
}

PFGSFixLinker::~PFGSFixLinker()
{
}

void
PFGSFixLinker::produce(edm::Event& _event, edm::EventSetup const&)
{
  std::auto_ptr<reco::PFCandidateCollection> pOutput(new reco::PFCandidateCollection);

  auto& inCandidates(*getHandle(_event, inputCandidatesToken_, "PFCandidate"));
  auto&& inElectronsHandle(getHandle(_event, inputElectronsToken_, "GsfElectrons"));
  auto& electronMap(*getHandle(_event, electronMapToken_, "GsfElectronsMap"));
  auto&& inPhotonsHandle(getHandle(_event, inputPhotonsToken_, "Photons"));
  auto& photonMap(*getHandle(_event, photonMapToken_, "PhotonsMap"));

  std::vector<reco::PFCandidatePtr> oldPtrs;
  std::vector<unsigned> pfElectronIndices(inElectronsHandle->size(), -1);
  std::vector<unsigned> pfPhotonIndices(inPhotonsHandle->size(), -1);

  unsigned iP(0);
  for (auto& inCand : inCandidates) {
    oldPtrs.emplace_back(inCandidates.ptrAt(iP++));

    pOutput->emplace_back(inCand);
    auto& outCand(pOutput->back());
    
    auto&& eRef(inCand.gsfElectronRef());
    if (eRef.isNonnull()) {
      auto&& newERef(GainSwitchTools::findNewRef(eRef, inElectronsHandle, electronMap));
      auto& newE(*newERef);
      auto& newSC(*newE.superCluster());

      pfElectronIndices[newERef.key()] = pOutput->size() - 1;

      outCand.setGsfElectronRef(newERef);
      outCand.setSuperClusterRef(newE.superCluster());
      outCand.setEcalEnergy(newSC.rawEnergy(), newE.ecalEnergy());
      outCand.setDeltaP(newE.p4Error(reco::GsfElectron::P4_COMBINATION));
      outCand.setP4(newE.p4(reco::GsfElectron::P4_COMBINATION));
    }

    auto&& phRef(inCand.photonRef());
    if (phRef.isNonnull()) {
      auto&& newPhRef(GainSwitchTools::findNewRef(phRef, inPhotonsHandle, photonMap));
      auto& newPh(*newPhRef);
      auto& newSC(*newPh.superCluster());

      pfPhotonIndices[newPhRef.key()] = pOutput->size() - 1;
    
      outCand.setPhotonRef(newPhRef);
      outCand.setSuperClusterRef(newPh.superCluster());
      outCand.setEcalEnergy(newSC.rawEnergy(), newPh.getCorrectedEnergy(reco::Photon::regression2));
      outCand.setDeltaP(newPh.getCorrectedEnergyError(reco::Photon::regression2));
      outCand.setP4(newPh.p4(reco::Photon::regression2));
    }
  }

  auto&& outCandsHandle(_event.put(pOutput));

  std::auto_ptr<PFPtrMap> pPFMap(new PFPtrMap);
  PFPtrMap::Filler pfMapFiller(*pPFMap);
  pfMapFiller.insert(outCandsHandle, oldPtrs.begin(), oldPtrs.end());
  pfMapFiller.fill();
  _event.put(pPFMap);

  std::vector<reco::PFCandidatePtr> pfElectrons;
  for (unsigned idx : pfElectronIndices) {
    if (idx == unsigned(-1))
      pfElectrons.emplace_back();
    else
      pfElectrons.emplace_back(outCandsHandle, idx);
  }

  std::auto_ptr<PFPtrMap> pEleMap(new PFPtrMap);
  PFPtrMap::Filler eleMapFiller(*pEleMap);
  eleMapFiller.insert(inElectronsHandle, pfElectrons.begin(), pfElectrons.end());
  eleMapFiller.fill();
  _event.put(pEleMap, electronsMapName_);

  std::vector<reco::PFCandidatePtr> pfPhotons;
  for (unsigned idx : pfPhotonIndices) {
    if (idx == unsigned(-1))
      pfPhotons.emplace_back();
    else
      pfPhotons.emplace_back(outCandsHandle, idx);
  }

  std::auto_ptr<PFPtrMap> pPhoMap(new PFPtrMap);
  PFPtrMap::Filler phoMapFiller(*pPhoMap);
  phoMapFiller.insert(inPhotonsHandle, pfPhotons.begin(), pfPhotons.end());
  phoMapFiller.fill();
  _event.put(pPhoMap, photonsMapName_);
}

DEFINE_FWK_MODULE(PFGSFixLinker);
