#include <memory>

#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonCoreGSCrysFixer.h"

#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

GEDPhotonCoreGSCrysFixer::GEDPhotonCoreGSCrysFixer(const edm::ParameterSet& config)
{
  getToken(inputCoresToken_, config, "photonCores");
  getToken(refinedSCsToken_, config, "refinedSCs");
  getToken(refinedSCMapToken_, config, "refinedSCs");
  getToken(ebSCsToken_, config, "scs", "particleFlowSuperClusterECALBarrel");
  getToken(ebSCMapToken_, config, "refinedSCs", "parentSCsEB");
  getToken(eeSCsToken_, config, "scs", "particleFlowSuperClusterECALEndcapWithPreshower");
  getToken(eeSCMapToken_, config, "refinedSCs", "parentSCsEE");
  getToken(convsToken_, config, "conversions");
  getToken(singleLegConvsToken_, config, "singleconversions");

  produces<reco::PhotonCoreCollection>();
  produces<SCRefMap>(); // new core to old SC
}

GEDPhotonCoreGSCrysFixer::~GEDPhotonCoreGSCrysFixer()
{
}

void
GEDPhotonCoreGSCrysFixer::produce(edm::Event& _event, edm::EventSetup const&)
{
  std::auto_ptr<reco::PhotonCoreCollection> pOutput(new reco::PhotonCoreCollection);

  auto& inputCores(*getHandle(_event, inputCoresToken_, "photonCores"));
  auto refinedSCs(getHandle(_event, refinedSCsToken_, "refinedSCs"));
  auto& refinedSCMap(*getHandle(_event, refinedSCMapToken_, "refinedSCMap"));
  auto ebSCs(getHandle(_event, ebSCsToken_, "ebSCs"));
  auto& ebSCMap(*getHandle(_event, ebSCMapToken_, "ebSCMap"));
  auto eeSCs(getHandle(_event, eeSCsToken_, "eeSCs"));
  auto& eeSCMap(*getHandle(_event, eeSCMapToken_, "eeSCMap"));
  auto convs(getHandle(_event, convsToken_, "conversions"));
  auto singleLegConvs(getHandle(_event, singleLegConvsToken_, "singleLegConversions"));

  std::vector<reco::SuperClusterRef> oldSCRefs;

  for (auto& inCore : inputCores) {
    pOutput->emplace_back();
    auto& outCore(pOutput->back());

    // NOTE: These mappings can result in NULL superclusters!
    auto oldRefinedSC(inCore.superCluster());
    EBDetId seedId(oldRefinedSC->seed()->seed());

    outCore.setSuperCluster(GainSwitchTools::findNewRef(oldRefinedSC, refinedSCs, refinedSCMap));

    oldSCRefs.push_back(oldRefinedSC);

    auto parentSC(inCore.parentSuperCluster());
    if (parentSC.isNonnull()) {
      if (parentSC->seed()->seed().subdetId() == EcalBarrel)
        outCore.setParentSuperCluster(GainSwitchTools::findNewRef(parentSC, ebSCs, ebSCMap));
      else
        outCore.setParentSuperCluster(GainSwitchTools::findNewRef(parentSC, eeSCs, eeSCMap));
    }

    outCore.setPFlowPhoton(true);
    outCore.setStandardPhoton(false);

    // here we rely on ConversionGSCrysFixer and EGRefinedSCFixer translating conversions in the same order as the input
    // super ugly and not safe but we don't have a dictionary for ValueMap<ConversionRef>

    for (auto&& oldPtr : inCore.conversions())
      outCore.addConversion(reco::ConversionRef(convs, oldPtr.key()));
    
    for (auto&& oldPtr : inCore.conversionsOneLeg())
      outCore.addOneLegConversion(reco::ConversionRef(singleLegConvs, oldPtr.key()));

    for (auto&& seed : inCore.electronPixelSeeds())
      outCore.addElectronPixelSeed(seed);
  }

  auto newCoresHandle(_event.put(pOutput));

  std::auto_ptr<SCRefMap> pRefMap(new SCRefMap);
  SCRefMap::Filler refMapFiller(*pRefMap);
  refMapFiller.insert(newCoresHandle, oldSCRefs.begin(), oldSCRefs.end());
  refMapFiller.fill();
  _event.put(pRefMap);
}
