#include <map>
#include <string>

#include "TH1.h"
#include "PhysicsTools/UtilAlgos/interface/BasicAnalyzer.h"
#include "DataFormats/MuonReco/interface/Muon.h"

/**
   \class BasicMuonAnalyzer BasicMuonAnalyzer.h "PhysicsTools/UtilAlgos/interface/BasicMuonAnalyzer.h"
   \brief Example class that can be used both within FWLite and within the full framework

   This is an example for keeping classes that can be used both within FWLite and within the full
   framework. The class is derived from the BasicAnalyzer base class, which is an interface for
   the two wrapper classes EDAnalyzerWrapper and FWLiteAnalyzerWrapper. The latter provides basic
   configuration file reading and event looping equivalent to the FWLiteHistograms executable of
   this package. You can see the FWLiteAnalyzerWrapper class at work in the FWLiteWithBasicAnalyzer
   executable of this package.
*/

class BasicMuonAnalyzer : public edm::BasicAnalyzer {

 public:
  /// default constructor
  BasicMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs);
  BasicMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs, edm::ConsumesCollector&& iC);
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
  edm::EDGetTokenT<std::vector<reco::Muon> > muonsToken_;
  /// histograms
  std::map<std::string, TH1*> hists_;
};
