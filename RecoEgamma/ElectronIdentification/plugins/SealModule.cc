#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"

//#include "RecoEgamma/ElectronIdentification/interface/ElectronIDProducer.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelector.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorNeuralNet.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorLikelihood.h"

typedef ElectronIDSelector<ElectronIDSelectorCutBased>   EleIdCutBasedSel;
typedef ElectronIDSelector<ElectronIDSelectorNeuralNet>  EleIdNeuralNetSel;
typedef ElectronIDSelector<ElectronIDSelectorLikelihood> EleIdLikelihoodSel;
//typedef ObjectSelector<EleIdCutBasedSel> EleIdCutBased ;
typedef ObjectSelector<
          EleIdCutBasedSel, 
          edm::RefVector<reco::PixelMatchGsfElectronCollection> 
         > EleIdCutBasedRef ;
typedef ObjectSelector<
          EleIdNeuralNetSel, 
          edm::RefVector<reco::PixelMatchGsfElectronCollection> 
         > EleIdNeuralNetRef ;
typedef ObjectSelector<
          EleIdLikelihoodSel, 
          edm::RefVector<reco::PixelMatchGsfElectronCollection> 
         > EleIdLikelihoodRef ;


DEFINE_SEAL_MODULE();

//DEFINE_ANOTHER_FWK_MODULE(ElectronIDProducer);

//DEFINE_ANOTHER_FWK_MODULE(EleIdCutBased);
DEFINE_ANOTHER_FWK_MODULE(EleIdCutBasedRef);
DEFINE_ANOTHER_FWK_MODULE(EleIdNeuralNetRef);
DEFINE_ANOTHER_FWK_MODULE(EleIdLikelihoodRef);
