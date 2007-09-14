#ifndef HLTriggerOffline_Muon_HLTMuonTime_H
#define HLTriggerOffline_Muon_HLTMuonTime_H

/** \class HLTMuonTime
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author  M. Vander Donckt  (copied fromJ. Alcaraz
 */

// Base Class Headers

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "TDirectory.h"

class TH1F;

class HLTMuonTime {
public:
  /// Constructor
  HLTMuonTime(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~HLTMuonTime();

  // Operations

  void analyze(const edm::Event & event);

  virtual void BookHistograms() ;
  virtual void CreateHistograms(std::string type, std::string module) ;
  virtual void CreateGlobalHistograms(std::string name, std::string title) ;
  virtual void WriteHistograms() ;

private:
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
  std::vector <TH1F*> hTimes;
  std::vector <TH1F*> hGlobalTimes;
  std::vector <TH1F*> hExclusiveTimes;
  std::vector <TH1F*> hExclusiveGlobalTimes;
  std::vector <std::string> ModuleNames;
  std::vector <double> ModuleTime;
  std::vector <int> NumberOfModules;
  std::vector <TDirectory *> TDirs;
  int theNbins;
  double theTMax;
  TDirectory* HistoDir;
  TDirectory* muondigi;
  TDirectory* trackerdigi;
  TDirectory* trackerrec;
  TDirectory* calodigi;
  TDirectory* calorec;
  TDirectory* muonlocrec;
  TDirectory* muonl2rec;
  TDirectory* muonl2iso;
  TDirectory* muonl3rec;
  TDirectory* muonl3iso;
  
};
#endif
