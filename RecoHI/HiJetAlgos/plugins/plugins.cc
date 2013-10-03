#include "RecoHI/HiJetAlgos/interface/MultipleAlgoIterator.h"
#include "RecoHI/HiJetAlgos/interface/JetOffsetCorrector.h"
#include "RecoHI/HiJetAlgos/interface/ParametrizedSubtractor.h"
#include "RecoHI/HiJetAlgos/interface/ReflectedIterator.h"
#include "RecoHI/HiJetAlgos/interface/VoronoiSubtractor.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,JetOffsetCorrector,"JetOffsetCorrector");
DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,MultipleAlgoIterator,"MultipleAlgoIterator");
DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,ParametrizedSubtractor,"ParametrizedSubtractor");
DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,ReflectedIterator,"ReflectedIterator");
DEFINE_EDM_PLUGIN(PileUpSubtractorFactory,VoronoiSubtractor,"VoronoiSubtractor");





