#include <memory>
/** \class ConversionGSCrysFixer
 **
 **
 **  \author Yutaro Iiyama, MIT
 **
 ***/

#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionGSCrysFixer.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

ConversionGSCrysFixer::ConversionGSCrysFixer(const edm::ParameterSet& config)
{
  getToken(inputConvsToken_, config, "conversions");
  getToken(ebSCsToken_, config, "superClusters", "particleFlowSuperClusterECALBarrel");
  getToken(ebSCMapToken_, config, "scMaps", "parentSCsEB");
  getToken(eeSCsToken_, config, "superClusters", "particleFlowSuperClusterECALEndcapWithPreshower");
  getToken(eeSCMapToken_, config, "scMaps", "parentSCsEE");

  produces<reco::ConversionCollection>();
}

ConversionGSCrysFixer::~ConversionGSCrysFixer()
{
}

void
ConversionGSCrysFixer::produce(edm::Event& _event, edm::EventSetup const&)
{
  std::auto_ptr<reco::ConversionCollection> pOutput(new reco::ConversionCollection);

  auto& inputConvs(*getHandle(_event, inputConvsToken_, "conversions"));
  auto ebSCs(getHandle(_event, ebSCsToken_, "ebSCs"));
  auto& ebSCMap(*getHandle(_event, ebSCMapToken_, "ebSCMap"));
  auto eeSCs(getHandle(_event, eeSCsToken_, "eeSCs"));
  auto& eeSCMap(*getHandle(_event, eeSCMapToken_, "eeSCMap"));

  for (auto& inConv : inputConvs) {
    pOutput->emplace_back(inConv);
    auto& newConv(pOutput->back());

    reco::CaloClusterPtrVector clusters;
    for (auto ptr : inConv.caloCluster()) {
      auto* sc(static_cast<reco::SuperCluster const*>(ptr.get()));
      reco::SuperClusterRef newSC;
      if (sc->seed()->seed().subdetId() == EcalBarrel)
        newSC = GainSwitchTools::findNewRef(ptr, ebSCs, ebSCMap);
      else
        newSC = GainSwitchTools::findNewRef(ptr, eeSCs, eeSCMap);

      if (newSC.isNonnull())
        clusters.push_back(edm::refToPtr(newSC));
    }

    newConv.setMatchingSuperCluster(clusters);
  }

  _event.put(pOutput);
}
