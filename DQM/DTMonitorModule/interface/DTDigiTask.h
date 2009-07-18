#ifndef DTDigiTask_H
#define DTDigiTask_H

/*
 * \file DTDigiTask.h
 *
 * $Date: 2009/02/26 12:08:47 $
 * $Revision: 1.29 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include <FWCore/Framework/interface/LuminosityBlock.h>
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTSuperLayerId;
class DTLayerId;
class DTChamberId;
class DTTtrig;
class DTT0;

class DQMStore;
class MonitorElement;

class DTDigiTask: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDigiTask(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTDigiTask();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  void beginRun(const edm::Run&, const edm::EventSetup&);

  /// Book the ME
  void bookHistos(const DTSuperLayerId& dtSL, std::string folder, std::string histoTag);
  void bookHistos(const DTChamberId& dtCh, std::string folder, std::string histoTag);
  void bookHistos(const int wheelId, std::string folder, std::string histoTag);

  /// To reset the MEs
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& setup);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// get the L1A source
  std::string triggerSource();

private:
  
  std::string topFolder() const;

  int nevents;

  /// no needs to be precise. Value from PSets will always be used
  int tMax;
  int maxTDCHits;

  /// tTrig from the DB
  float tTrig;
  float tTrigRMS;
  float kFactor;

  //check for sync noise
  //  bool newChamber;
  //  DTChamberId chDone;
  std::map<DTChamberId,int> hitMap;
  std::set<DTChamberId> syncNoisyChambers;
  int syncNumTot;
  int syncNum;

  edm::Handle<LTCDigiCollection> ltcdigis;

  DQMStore* dbe;

  edm::ESHandle<DTGeometry> muonGeom;

  edm::ESHandle<DTTtrig> tTrigMap;
  edm::ESHandle<DTT0> t0Map;

  std::map<std::string, std::map<uint32_t, MonitorElement*> > digiHistos;
  std::map<std::string, std::map<int, MonitorElement*> > wheelHistos;

  // Parameters from config file

  // The label to retrieve the digis 
  edm::InputTag dtDigiLabel;
  // Set to true to read the ttrig from DB (useful to determine in-time and out-of-time hits)
  bool readTTrigDB;
  // Set to true to subtract t0 from test pulses
  bool subtractT0;
  // Tmax value (TDC counts)
  int defaultTmax;
  // Switch from static (all histo at the beginninig of the job) to
  // dynamic (book when needed) histo booking
  bool doStaticBooking;
  // Switch for local/global runs
  bool isLocalRun;
  // Setting for the reset of the ME after n (= ResetCycle) luminosity sections
  int resetCycle;
  // Check the DB of noisy channels
  bool checkNoisyChannels;
  // Default TTrig to be used when not reading the TTrig DB
  int defaultTTrig;

  int inTimeHitsLowerBound;
  int inTimeHitsUpperBound;
  int timeBoxGranularity;
  int maxTDCCounts;
  bool doAllHitsOccupancies;
  bool doNoiseOccupancies;
  bool doInTimeOccupancies;

  bool tpMode;
  bool lookForSyncNoise;
  bool filterSyncNoise;

  bool doLayerTimeBoxes;

  std::map<DTChamberId, int> nSynchNoiseEvents;
  MonitorElement* nEventMonitor;



};

#endif
