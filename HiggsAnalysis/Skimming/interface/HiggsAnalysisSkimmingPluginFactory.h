#ifndef HiggsAnalysisSkimming_HiggsAnalysisSkimmingPluginFactory
#define HiggsAnalysisSkimming_HiggsAnalysisSkimmingPluginFactory

/** \class HiggsAnalysisSkimmingPluginFactory
 *
 *  Plugin factory for concrete HiggsAnalysisSkimming skim classes
 *
 *  \author Dominique Fortin - UC Riverside
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimType.h>

class edm::ParameterSet;

typedef edmplugin::PluginFactory<HiggsAnalysisSkimType *(const edm::ParameterSet&)> HiggsAnalysisSkimmingPluginFactory;

#endif
