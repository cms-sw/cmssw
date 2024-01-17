/* \class ElectronCollectionMerger
 *
 * Producer of merged Electron collection
 *
 * \author: Michal Bluj, NCBJ, Poland
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

typedef Merger<reco::GsfElectronCollection> GsfElectronCollectionMerger;
template <>
void GsfElectronCollectionMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("gsfElectronCollectionMerger", desc);
}
DEFINE_FWK_MODULE(GsfElectronCollectionMerger);

typedef Merger<pat::ElectronCollection> PATElectronCollectionMerger;
template <>
void PATElectronCollectionMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("patElectronCollectionMerger", desc);
}
DEFINE_FWK_MODULE(PATElectronCollectionMerger);
