#ifndef DTLocalTriggerSynchTask_H
#define DTLocalTriggerSynchTask_H

/*
 * \file DTLocalTriggerSynchTask.h
 *
 * $Date: 2010/03/05 09:11:52 $
 * $Revision: 1.4 $
 * \author C. Battilana - CIEMAT
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DTRecSegment4D;
class DTTTrigBaseSync;
class DTLocalTrigger;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;


class DTLocalTriggerSynchTask: public edm::EDAnalyzer{
  
  friend class DTMonitorModule;
  
 public:
  
  /// Constructor
  DTLocalTriggerSynchTask(const edm::ParameterSet& ps );
  
  /// Destructor
  virtual ~DTLocalTriggerSynchTask();
  
 protected:
  
  // BeginJob
  void beginJob();
  
  /// Book the histograms
  void bookHistos(const DTChamberId& dtCh );
  
  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context);

  /// Begin Run
  void beginRun(const edm::Run& run, const edm::EventSetup& context);
  
  /// EndJob
  void endJob(void);

  std::string & baseDir() { return baseDirectory; }
    
 private:
  
  int nevents;
 
  int phCodeBestDCC[6][5][13];
  int phCodeBXDCC[6][5][13][3];
  int phCodeBestDDU[6][5][13];
  int thCodeBestDDU[6][5][13];
  int segHitBest[6][5][13];

  float bxTime;
  bool rangeInBX;
  int nBXLow;
  int nBXHigh;
  float angleRange;
  float minHitsPhi;
  int fineDelay;
  DTTTrigBaseSync *tTrigSync;

  std::string baseDirectory;

  DQMStore* dbe;
  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > triggerHistos;
  MonitorElement* dcc_IDDataErrorPlot;

};

#endif
