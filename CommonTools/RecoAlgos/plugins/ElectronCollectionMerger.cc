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
DEFINE_FWK_MODULE(GsfElectronCollectionMerger);

typedef Merger<pat::ElectronCollection> PATElectronCollectionMerger;
DEFINE_FWK_MODULE(PATElectronCollectionMerger);
