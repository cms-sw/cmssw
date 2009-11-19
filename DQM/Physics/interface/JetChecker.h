#ifndef Jet_Checker_h
#define Jet_Checker_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

/**
   \class   JetChecker JetChecker.h "DQM/Physics/interface/JetChecker.h"

   \brief   class to fill monitor histograms for jets

   Add a more detailed description here...
*/

class JetChecker  {

 public:
  /// default constructor
  explicit JetChecker(const edm::ParameterSet& cfg, const std::string& directory, const std::string& label);
  /// default destructor
  ~JetChecker();

  /// everything that needs to be done before the event loop
  void begin(const edm::EventSetup& setup, const std::string& corrector);
  /// everything that needs to be done during the event loop
  void analyze(const std::vector<reco::CaloJet>& jets,  bool useJES, const edm::Event& event, const edm::EventSetup& setup);
  /// everything that needs to be done during the event loop
  void analyzeWithBjets(const std::vector<reco::CaloJet>&jets, bool useJES, const edm::Event& event, const edm::EventSetup& setup);
  /// everything that needs to be done after the event loop
  void end() ;

 private:
  //  b-tagging fill methods:
  void beginJobBtagging(const edm::EventSetup& setup);
  
 private:
  /// use for the name of the directory
  std::string relativePath_;
  /// use for the name of the directory
  std::string label_; 
  /// number of bins for several histograms 
  int nBins_;
  /// dqm storage element
  DQMStore* dqmStore_;
  /// histogram container
  std::map<std::string,MonitorElement*> hists_;
  /// jet corrector
  const JetCorrector* corrector ;
  /// ...
  bool checkBtaggingSet_;   // bookkeeping bool to cross check that this is only done if btagging bool set correctly.
  /// ...
  std::vector<std::string> btaggingalgonames_; // vector with all algorithm names
  /// ...
  std::string makeBtagHistName(const size_t & index);
  /// ...
  std::string makeBtagCutHistName(const size_t &index);
  /// ...
  double btaggingMatchDr_;
  /// ...
  edm::ParameterSet btaggingcuts_;
  /// ...
  std::vector<std::pair<std::string,double > > workingpoints_;
};

#endif
