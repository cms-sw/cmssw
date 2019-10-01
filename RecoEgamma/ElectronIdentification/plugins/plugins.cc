#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedIdProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelector.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"

typedef VersionedIdProducer<reco::GsfElectronPtr> VersionedGsfElectronIdProducer;
DEFINE_FWK_MODULE(VersionedGsfElectronIdProducer);

typedef VersionedIdProducer<pat::ElectronPtr> VersionedPatElectronIdProducer;
DEFINE_FWK_MODULE(VersionedPatElectronIdProducer);

typedef ElectronIDSelector<ElectronIDSelectorCutBased> EleIdCutBasedSel;

#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDExternalProducer.h"
typedef ElectronIDExternalProducer<ElectronIDSelectorCutBased> EleIdCutBasedExtProducer;
DEFINE_FWK_MODULE(EleIdCutBasedExtProducer);

typedef ObjectSelector<EleIdCutBasedSel, reco::GsfElectronCollection> EleIdCutBased;
DEFINE_FWK_MODULE(EleIdCutBased);

#include "RecoEgamma/EgammaTools/interface/MVAValueMapProducer.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

typedef MVAValueMapProducer<reco::GsfElectron> ElectronMVAValueMapProducer;
DEFINE_FWK_MODULE(ElectronMVAValueMapProducer);

#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory, ElectronMVAEstimatorRun2, "ElectronMVAEstimatorRun2");
