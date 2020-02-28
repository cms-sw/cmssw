#include <map>
#include <string>

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TH1.h"
#include "PhysicsTools/UtilAlgos/interface/BasicAnalyzer.h"

/**
   \class AnalysisTasksAnalyzerBTag AnalysisTasksAnalyzerBTag.h "PhysicsTools/UtilAlgos/interface/AnalysisTasksAnalyzerBTag.h"
   \brief Example class that can be used both within FWLite and within the full framework

   This is an example for keeping classes that can be used both within FWLite and within the full
   framework. The class is derived from the BasicAnalyzer base class, which is an interface for
   the two wrapper classes EDAnalyzerWrapper and FWLiteAnalyzerWrapper. The latter provides basic
   configuration file reading and event looping equivalent to the FWLiteHistograms executable of
   this package. You can see the FWLiteAnalyzerWrapper class at work in the FWLiteWithBasicAnalyzer
   executable of this package.
*/

class AnalysisTasksAnalyzerBTag : public edm::BasicAnalyzer {
public:
  /// default constructor
  AnalysisTasksAnalyzerBTag(const edm::ParameterSet& cfg, TFileDirectory& fs);
  AnalysisTasksAnalyzerBTag(const edm::ParameterSet& cfg, TFileDirectory& fs, edm::ConsumesCollector&& iC);
  /// default destructor
  ~AnalysisTasksAnalyzerBTag() override;
  /// everything that needs to be done before the event loop
  void beginJob() override{};
  /// everything that needs to be done after the event loop
  void endJob() override{};
  /// everything that needs to be done during the event loop
  void analyze(const edm::EventBase& event) override;

private:
  /// input tag for mouns
  edm::InputTag Jets_;
  edm::EDGetTokenT<std::vector<pat::Jet> > JetsToken_;
  std::string bTagAlgo_;
  unsigned int bins_;
  double lowerbin_;
  double upperbin_;
  /// histograms
  std::map<std::string, TH1*> hists_;
};
