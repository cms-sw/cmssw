#ifndef HLTriggerOffline_Muon_HLTMuonTime_H
#define HLTriggerOffline_Muon_HLTMuonTime_H

/** \class HLTMuonTime
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author  M. Vander Donckt  (copied fromJ. Alcaraz
 */

// Base Class Headers
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class TH1F;

class HLTMuonTime {
public:
  /// Constructor
  HLTMuonTime(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~HLTMuonTime();

  // Operations

  void analyze(const edm::Event & event);

  void BookHistograms() ;
  void WriteHistograms() ;
  void CreateHistograms(std::string type, std::string module) ;
  void CreateGlobalHistograms(std::string name, std::string title) ;


private:
  DQMStore* dbe;  
  bool TimerIn;
  // Input from cfg file
  std::vector<std::string> theMuonDigiModules;
  std::vector<std::string> theTrackerDigiModules;
  std::vector<std::string> theTrackerRecModules;
  std::vector<std::string> theTrackerTrackModules;
  std::vector<std::string> theCaloDigiModules;
  std::vector<std::string> theCaloRecModules;
  std::vector<std::string> theMuonLocalRecModules;
  std::vector<std::string> theMuonL2RecModules;
  std::vector<std::string> theMuonL2IsoModules;
  std::vector<std::string> theMuonL3RecModules;
  std::vector<std::string> theMuonL3IsoModules;
  // Histograms
  std::vector <MonitorElement*> hTimes;
  std::vector <MonitorElement*> hGlobalTimes;
  std::vector <MonitorElement*> hExclusiveTimes;
  std::vector <MonitorElement*> hExclusiveGlobalTimes;
  std::vector <std::string> ModuleNames;
  std::vector <double> ModuleTime;
  std::vector <int> NumberOfModules;
  std::vector <std::string> TDirs;
  int theNbins;
  double theTMax;
  edm::InputTag theTimerLabel;
  std::string theRootFileName;

  
};
#endif
