// This is HiggsAnalysisSkimmingPluginFactory.cc

#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimmingPluginFactory.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

HiggsAnalysisSkimmingPluginFactory HiggsAnalysisSkimmingPluginFactory::s_instance;

HiggsAnalysisSkimmingPluginFactory::HiggsAnalysisSkimmingPluginFactory () :
  seal::PluginFactory<HiggsAnalysisSkimType *(const edm::ParameterSet&)>("HiggsAnalysisSkimmingPluginFactory"){}

HiggsAnalysisSkimmingPluginFactory* HiggsAnalysisSkimmingPluginFactory::get (){
  return &s_instance; 
}

