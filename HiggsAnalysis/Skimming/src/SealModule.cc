#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimming.h>
#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimmingPluginFactory.h>

#include <HiggsAnalysis/Skimming/interface/HiggsToXexampleSkim.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkim.h>

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HiggsAnalysisSkimming);
DEFINE_SEAL_PLUGIN(HiggsAnalysisSkimmingPluginFactory, HiggsToXexampleSkim, "HiggsToXexampleSkim");
DEFINE_SEAL_PLUGIN(HiggsAnalysisSkimmingPluginFactory, HiggsToZZ4LeptonsSkim, "HiggsToZZ4LeptonsSkim");

