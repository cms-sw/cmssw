#ifndef Muon_Checker_h
#define Muon_Checker_h

// system include files
#include <memory>
#include <string>
#include <vector>
#include "TLorentzVector.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//needed for MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"

/**
   \class   MetChecker MetChecker.h "DQM/Physics/interface/MetChecker.h"

   \brief   Add a one sentence description here...

   Module dedicated to muons
   It takes a vector as input instead of edm::View or edm::Handle
   -> possibility to have a selected collection
   It's based on EDAnalyzer
   It uses DQMStore to store the histograms
   in a directory: relativePath_+"Muons_"+label_
   This module has to be called by a channel-specific module like LeptonJetsChecker
*/

class MuonChecker{
 public:
  /// default constructor
  explicit MuonChecker(const edm::ParameterSet&, std::string relativePath, std::string label);
  /// default destructor
  ~MuonChecker();

  /// everything that needs to be done before the event loop
  void begin(const edm::EventSetup& setup) ;
  /// everything that needs to be done during the event loop
  void analyze(const std::vector<reco::Muon>& muons);
  /// everything that needs to be done after the event loop
  void end() ;
  
 private:
  /// use for the name of the directory
  std::string relativePath_;
  /// use for the name of the directory
  std::string label_;
  /// dqm storage element  
  DQMStore* dqmStore_;
  /// histogram container  
  std::map<std::string,MonitorElement*> hists_;
};

#endif
