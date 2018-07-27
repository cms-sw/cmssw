#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelector.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorLikelihood.h"
typedef ElectronIDSelector<ElectronIDSelectorCutBased>   EleIdCutBasedSel;
typedef ElectronIDSelector<ElectronIDSelectorLikelihood> EleIdLikelihoodSel;
typedef ObjectSelector<
          EleIdCutBasedSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdCutBasedRef ;
typedef ObjectSelector<
          EleIdLikelihoodSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdLikelihoodRef ;
DEFINE_FWK_MODULE(EleIdCutBasedRef);
DEFINE_FWK_MODULE(EleIdLikelihoodRef);

#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDExternalProducer.h"
typedef ElectronIDExternalProducer<ElectronIDSelectorCutBased>   EleIdCutBasedExtProducer;
typedef ElectronIDExternalProducer<ElectronIDSelectorLikelihood> EleIdLikelihoodExtProducer;
DEFINE_FWK_MODULE(EleIdCutBasedExtProducer);
DEFINE_FWK_MODULE(EleIdLikelihoodExtProducer);

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronLikelihoodESSource.h"
DEFINE_FWK_EVENTSETUP_MODULE( ElectronLikelihoodESSource );

typedef ObjectSelector<EleIdCutBasedSel, reco::GsfElectronCollection> EleIdCutBased;
DEFINE_FWK_MODULE(EleIdCutBased);
