#ifndef DTLocalTriggerTask_H
#define DTLocalTriggerTask_H

/*
 * \file DTLocalTriggerTask.h
 *
 * $Date: 2007/04/02 16:19:50 $
 * $Revision: 1.4 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;

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

  /// Convert phi to local x coordinate
  float phi2Pos(const DTChamberId & id, int phi);

  /// Convert phib to global angle coordinate
  float phib2Ang(const DTChamberId & id, int phib, double phi);
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  /// EndJob
  void endJob(void);
  
  /// Get the L1A source
  std::string triggerSource();
  
 private:
  
  bool debug;
  std::string dcc_label;
  std::string ros_label;
  std::string seg_label;
  std::string outputFile;  
  int nevents;
 
  DaqMonitorBEInterface* dbe;
  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::Handle<LTCDigiCollection> ltcdigis;
  std::map<std::string, std::map<uint32_t, MonitorElement*> > digiHistos;
  
};

#endif
