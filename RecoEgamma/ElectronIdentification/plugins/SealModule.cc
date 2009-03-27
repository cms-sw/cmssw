#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"

#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelector.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorNeuralNet.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorLikelihood.h"
typedef ElectronIDSelector<ElectronIDSelectorCutBased>   EleIdCutBasedSel;
typedef ElectronIDSelector<ElectronIDSelectorNeuralNet>  EleIdNeuralNetSel;
typedef ElectronIDSelector<ElectronIDSelectorLikelihood> EleIdLikelihoodSel;
typedef ObjectSelector<
          EleIdCutBasedSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdCutBasedRef ;
typedef ObjectSelector<
          EleIdNeuralNetSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdNeuralNetRef ;
typedef ObjectSelector<
          EleIdLikelihoodSel, 
          edm::RefVector<reco::GsfElectronCollection> 
         > EleIdLikelihoodRef ;
DEFINE_FWK_MODULE(EleIdCutBasedRef);
DEFINE_ANOTHER_FWK_MODULE(EleIdNeuralNetRef);
DEFINE_ANOTHER_FWK_MODULE(EleIdLikelihoodRef);

#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDExternalProducer.h"
typedef ElectronIDExternalProducer<ElectronIDSelectorCutBased>   EleIdCutBasedExtProducer;
typedef ElectronIDExternalProducer<ElectronIDSelectorNeuralNet>  EleIdNeuralNetExtProducer;
typedef ElectronIDExternalProducer<ElectronIDSelectorLikelihood> EleIdLikelihoodExtProducer;
DEFINE_ANOTHER_FWK_MODULE(EleIdCutBasedExtProducer);
DEFINE_ANOTHER_FWK_MODULE(EleIdNeuralNetExtProducer);
DEFINE_ANOTHER_FWK_MODULE(EleIdLikelihoodExtProducer);

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronLikelihoodESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE( ElectronLikelihoodESSource );
