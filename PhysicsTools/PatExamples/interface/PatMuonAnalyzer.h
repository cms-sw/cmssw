#include <map>
#include <string>

#include "TH1.h"
#include "PhysicsTools/UtilAlgos/interface/BasicAnalyzer.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

/**
   \class PatMuonAnalyzer PatMuonAnalyzer.h "PhysicsTools/PatExamples/interface/PatMuonAnalyzer.h"
   \brief Example class that can be used to analyze pat::Muons both within FWLite and within the full framework

   This is an example for keeping classes that can be used both within FWLite and within the full
   framework. The class is derived from the BasicAnalyzer base class, which is an interface for
   the two wrapper classes EDAnalyzerWrapper and FWLiteAnalyzerWrapper. You can fin more information
   on this on WorkBookFWLiteExamples#ExampleFive.
*/

class PatMuonAnalyzer : public edm::BasicAnalyzer {

 public:
  /// default constructor
  PatMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs);
  PatMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs, edm::ConsumesCollector&& iC);
  /// default destructor
  virtual ~PatMuonAnalyzer(){};
  /// everything that needs to be done before the event loop
  void beginJob(){};
  /// everything that needs to be done after the event loop
  void endJob(){};
  /// everything that needs to be done during the event loop
  void analyze(const edm::EventBase& event);

 private:
  /// input tag for mouns
  edm::InputTag muons_;
  edm::EDGetTokenT<std::vector<pat::Muon> > muonsToken_;
  /// histograms
  std::map<std::string, TH1*> hists_;
};
