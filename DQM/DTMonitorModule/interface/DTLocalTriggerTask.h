#ifndef DTLocalTriggerTask_H
#define DTLocalTriggerTask_H

/*
 * \file DTLocalTriggerTask.h
 *
 * $Date: 2007/11/12 18:00:30 $
 * $Revision: 1.13 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DTRecSegment4D;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;


class DTLocalTriggerTask: public edm::EDAnalyzer{
  
  friend class DTMonitorModule;
  
 public:
  
  /// Constructor
  DTLocalTriggerTask(const edm::ParameterSet& ps );
  
  /// Destructor
  virtual ~DTLocalTriggerTask();
  
 protected:
  
  // BeginJob
  void beginJob(const edm::EventSetup& c);
  
  /// Book the histograms
  void bookHistos(const DTChamberId& dtCh, std::string folder, std::string histoTag );
  
  /// Calculate phi range for histograms
  std::pair<float,float> phiRange(const DTChamberId& id);

  /// Compute track coordinates using trigger SC sectors
  void computeCoordinates(const DTRecSegment4D& track, int& scsector, float& phpos, float& phdir, float& zpos, float& zdir);

  /// Convert phi to local x coordinate
  float phi2Pos(const DTChamberId & id, int phi);

  /// Convert phib to global angle coordinate
  float phib2Ang(const DTChamberId & id, int phib, double phi); 

  /// Run analysis on DCC data
  void runDCCAnalysis(const edm::Event& e, std::string& trigsrc);

  /// Run analysis on ROS data
  void runDDUAnalysis(const edm::Event& e, std::string& trigsrc);

  /// Run analysis using DT 4D segments
  void runSegmentAnalysis(const edm::Event& e, std::string& trigsrc);

  /// Load DTTF map correction
  void loadDTTFMap();

  /// Correct DTTF mapping
  void correctMapping(int& wh, int& sector);
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
  
  /// EndJob
  void endJob(void);
  
  /// Get the L1A source
  std::string triggerSource(const edm::Event& e);
  
 private:
  
  bool debug;
  std::string dcc_label;
  std::string ros_label;
  std::string seg_label;
  std::string outputFile;  
  int nevents;
 
  int phcode_best[6][5][13];  
  int dduphcode_best[6][5][13];
  int thcode_best[6][5][13];  
  int dduthcode_best[6][5][13];
  int mapDTTF[6][13][2];
  const L1MuDTChambPhDigi* iphbest[6][5][13];
  const L1MuDTChambThDigi* ithbest[6][5][13];
  bool track_flag[6][5][15];
  bool track_ok[6][5][15];

  DaqMonitorBEInterface* dbe;
  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  std::map<std::string, std::map<uint32_t, MonitorElement*> > digiHistos;
  
  
};

#endif
