#include <map>
#include <string>
#include "TH1.h"

#include "PhysicsTools/UtilAlgos/interface/BasicAnalyzer.h"

/**
   \class BasicMuonAnalyzer BasicMuonAnalyzer.h "PhysicsTools/PatExamples/interface/BasicMuonAnalyzer.h"
   \brief Example class that can be used both within FWLite and within the ful framework

   This is an example for keeping classes that can be used both within FWLite and within the full framework. 
   The class is derived from the class BasicAnalyzer which is an interface for the two wrapper classes 
   EDAnalyzerWrapper and FWLiteAnalyzerWrapper. The latter provides basic configuration file reading and 
   event looping equivalent to the PatBasicFWliteAnalyzer example of this package. You can see both wrapper 
   classes at work in: 

   + PatExamples/bin/WrappedFWLiteMuonAnalyzer.cc
   + PatExamples/plugins/WrappedEDMuonAnalyzer.cc
*/

class BasicMuonAnalyzer : public edm::BasicAnalyzer {

 public:
  /// default constructor
  BasicMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs);
  /// default destructor
  virtual ~BasicMuonAnalyzer(){};
  /// everything that needs to be done before the event loop
  void beginJob(){};
  /// everything that needs to be done after the event loop
  void endJob(){};
  /// everything that needs to be done during the event loop
  void analyze(const edm::EventBase& event);

 private:
  /// input tag for mouns
  edm::InputTag muons_;
  /// histograms
  std::map<std::string, TH1*> hists_;
};
