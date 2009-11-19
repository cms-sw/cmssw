#ifndef Kinematics_h
#define Kinematics_h

// system include files
#include <memory>
#include <string>
#include <vector>
// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//needed for MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DataFormats
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"

/**
   \class   KinematicsChecker KinematicsChecker.h "DQM/Physics/interface/KinematicsChecker.h"

   \brief   class to fill monitor histograms for general kinematics

   Module dedicated to all objects: Electrons, Muons, CaloJets, CaloMets
   It takes a vector as input instead of edm::View or edm::Handle
   -> possibility to have a selected collection
   It uses DQMStore to store the histograms
   in this directories: 
   relativePath_+"Muons_"+label_
   relativePath_+"Electrons_"+label_
   relativePath_+"CaloJets_"+label_
   relativePath_+"CaloMETs_"+label_
   This module has to be called by a channel-specific module like LeptonJetsChecker
*/

struct Highest{
  bool operator()( double j1, double j2 ) const{
    return j1 > j2 ;
  }
};

class KinematicsChecker{

public:
  /// default constructor
  explicit KinematicsChecker(const edm::ParameterSet&, std::string relativePath, std::string label);
  /// default destructor
  ~KinematicsChecker();

  /// everything that needs to be done before the event loop
  void begin(const edm::EventSetup& setup);
  /// everything that needs to be done during the event loop
  void analyze(const std::vector<reco::CaloJet>& jets, const std::vector<reco::CaloMET>& mets, const std::vector<reco::Muon>& muons, const std::vector<reco::GsfElectron>& electrons);
  /// everything that needs to be done after the event loop
  void end();

private:
  /// ...
  int NbOfEvents;
  /// use for the name of the directory
  std::string relativePath_; 
  /// use for the name of the directory
  std::string label_; 
  /// dqm storage element
  DQMStore* dqmStore_;
  /// histogram container
  std::map<std::string,MonitorElement*> hists_[4];
};

#endif
