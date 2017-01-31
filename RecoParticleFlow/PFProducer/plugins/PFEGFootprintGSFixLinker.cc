#include "RecoParticleFlow/PFProducer/plugins/PFEGFootprintGSFixLinker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonEqual.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef edm::ValueMap<reco::PFCandidatePtr> PFPtrMap;

PFEGFootprintGSFixLinker::PFEGFootprintGSFixLinker(edm::ParameterSet const& _config) :
  electronsMapName_(_config.getParameter<std::string>("ValueMapElectrons")),
  photonsMapName_(_config.getParameter<std::string>("ValueMapPhotons"))
{
  getToken(newCandidatesToken_, _config, "PFCandidate");
  getToken(newElectronsToken_, _config, "GsfElectrons");
  getToken(newPhotonsToken_, _config, "Photons");
  getToken(electronMapToken_, _config, "GsfElectrons");
  getToken(photonMapToken_, _config, "Photons");
  getToken(electronFootprintMapToken_, _config, "GsfElectronsFootprint");
  getToken(photonFootprintMapToken_, _config, "PhotonsFootprint");
  
  // e/g collections to footprint PFs
  produces<FootprintMap>(electronsMapName_);
  produces<FootprintMap>(photonsMapName_);
}

PFEGFootprintGSFixLinker::~PFEGFootprintGSFixLinker()
{
}

void
PFEGFootprintGSFixLinker::produce(edm::Event& _event, edm::EventSetup const&)
{
  auto&& newCandidatesHandle(getHandle(_event, newCandidatesToken_, "PFCandidate"));
  auto&& newElectronsHandle(getHandle(_event, newElectronsToken_, "GsfElectrons"));
  auto& electronMap(*getHandle(_event, electronMapToken_, "GsfElectronsMap"));
  auto&& newPhotonsHandle(getHandle(_event, newPhotonsToken_, "Photons"));
  auto& photonMap(*getHandle(_event, photonMapToken_, "PhotonsMap"));
  auto& electronFootprintMap(*getHandle(_event, electronFootprintMapToken_, "GsfElectronsFootprint"));
  auto& photonFootprintMap(*getHandle(_event, photonFootprintMapToken_, "PhotonsFootprint"));

  std::vector<Footprint> electronFootprints;

  for (unsigned iE(0); iE != newElectronsHandle->size(); ++iE) {
    electronFootprints.emplace_back();
    auto& footprint(electronFootprints.back());

    reco::GsfElectronRef ref(newElectronsHandle, iE);
    auto& oldEleRef(electronMap[ref]);
    auto& oldFootprint(electronFootprintMap[oldEleRef]);
    // relying on PFGSFixLinker producing PF candidates in the same order
    for (auto& pfref : oldFootprint)
      footprint.emplace_back(newCandidatesHandle, pfref.key());
  }

  std::auto_ptr<FootprintMap> pEleFPMap(new FootprintMap);
  FootprintMap::Filler eleFPMapFiller(*pEleFPMap);
  eleFPMapFiller.insert(newElectronsHandle, electronFootprints.begin(), electronFootprints.end());
  eleFPMapFiller.fill();
  _event.put(pEleFPMap, electronsMapName_);

  std::vector<Footprint> photonFootprints;

  for (unsigned iE(0); iE != newPhotonsHandle->size(); ++iE) {
    photonFootprints.emplace_back();
    auto& footprint(photonFootprints.back());

    reco::PhotonRef ref(newPhotonsHandle, iE);
    auto& oldPhoRef(photonMap[ref]);
    auto& oldFootprint(photonFootprintMap[oldPhoRef]);
    // relying on PFGSFixLinker producing PF candidates in the same order
    for (auto& pfref : oldFootprint)
      footprint.emplace_back(newCandidatesHandle, pfref.key());
  }

  std::auto_ptr<FootprintMap> pPhoFPMap(new FootprintMap);
  FootprintMap::Filler phoFPMapFiller(*pPhoFPMap);
  phoFPMapFiller.insert(newPhotonsHandle, photonFootprints.begin(), photonFootprints.end());
  phoFPMapFiller.fill();
  _event.put(pPhoFPMap, photonsMapName_);
}

DEFINE_FWK_MODULE(PFEGFootprintGSFixLinker);
