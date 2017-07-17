#include "RecoHI/HiJetAlgos/interface/MultipleAlgoIterator.h"
#include "RecoHI/HiJetAlgos/interface/JetOffsetCorrector.h"
#include "RecoHI/HiJetAlgos/interface/ParametrizedSubtractor.h"
#include "RecoHI/HiJetAlgos/interface/ReflectedIterator.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,JetOffsetCorrector,"JetOffsetCorrector");
DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,MultipleAlgoIterator,"MultipleAlgoIterator");
DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,ParametrizedSubtractor,"ParametrizedSubtractor");
DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,ReflectedIterator,"ReflectedIterator");


#include "RecoHI/HiJetAlgos/interface/ParticleTowerProducer.h"
DEFINE_FWK_MODULE(ParticleTowerProducer);
#include "RecoHI/HiJetAlgos/interface/HiGenCleaner.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
typedef HiGenCleaner<reco::GenParticle> HiPartonCleaner;
typedef HiGenCleaner<reco::GenJet> HiGenJetCleaner;
DEFINE_FWK_MODULE(HiPartonCleaner);
DEFINE_FWK_MODULE(HiGenJetCleaner);





