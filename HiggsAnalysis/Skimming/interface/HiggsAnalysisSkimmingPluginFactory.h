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

class HiggsAnalysisSkimmingPluginFactory : public seal::PluginFactory<HiggsAnalysisSkimType *(const edm::ParameterSet&)>{
 public:
    /// Constructor
    HiggsAnalysisSkimmingPluginFactory();    

    static HiggsAnalysisSkimmingPluginFactory* get (void);

private:
    static HiggsAnalysisSkimmingPluginFactory s_instance;

};
#endif
