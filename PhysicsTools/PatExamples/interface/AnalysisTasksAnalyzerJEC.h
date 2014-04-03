#include <map>
#include <string>

#include "TH2.h"
#include "PhysicsTools/UtilAlgos/interface/BasicAnalyzer.h"

/**
   \class AnalysisTasksAnalyzerJEC AnalysisTasksAnalyzerJEC.h "PhysicsTools/UtilAlgos/interface/AnalysisTasksAnalyzerJEC.h"
   \brief Example class that can be used both within FWLite and within the full framework

   This is an example for keeping classes that can be used both within FWLite and within the full
   framework. The class is derived from the BasicAnalyzer base class, which is an interface for
   the two wrapper classes EDAnalyzerWrapper and FWLiteAnalyzerWrapper. The latter provides basic
   configuration file reading and event looping equivalent to the FWLiteHistograms executable of
   this package. You can see the FWLiteAnalyzerWrapper class at work in the FWLiteWithBasicAnalyzer
   executable of this package.
*/

class AnalysisTasksAnalyzerJEC : public edm::BasicAnalyzer {

 public:
  /// default constructor
  AnalysisTasksAnalyzerJEC(const edm::ParameterSet& cfg, TFileDirectory& fs);
  AnalysisTasksAnalyzerJEC(const edm::ParameterSet& cfg, TFileDirectory& fs, edm::ConsumesCollector&& iC);
  /// default destructor
  virtual ~AnalysisTasksAnalyzerJEC();
  /// everything that needs to be done before the event loop
  void beginJob(){};
  /// everything that needs to be done after the event loop
  void endJob(){};
  /// everything that needs to be done during the event loop
  void analyze(const edm::EventBase& event);

 private:
  /// input tag for mouns
  edm::InputTag Jets_;
  edm::EDGetTokenT<std::vector<pat::Jet> > JetsToken_;
  std::string jecLevel_;
  std::string patJetCorrFactors_;
  bool help_;
  unsigned int jetInEvents_;
  /// histograms
  std::map<std::string, TH2*> hists_;
};
