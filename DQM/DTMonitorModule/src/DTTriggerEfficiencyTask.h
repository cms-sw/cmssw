#ifndef DTTriggerEfficiencyTask_H
#define DTTriggerEfficiencyTask_H

/*
 * \file DTTriggerEfficiencyTask.h
 *
 * $Date: 2011/10/21 18:10:23 $
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

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DTTrigGeomUtils;

class DTTriggerEfficiencyTask: public edm::EDAnalyzer{
  
 public:
  
  /// Constructor
  DTTriggerEfficiencyTask(const edm::ParameterSet& ps );
  
  /// Destructor
  virtual ~DTTriggerEfficiencyTask();
  
 protected:
  
  // BeginJob
  void beginJob();
  
  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& context);

  /// Book chamber granularity histograms
  void bookChamberHistos(const DTChamberId& dtCh, std::string histoTag, std::string folder="");

  /// Book wheel granularity histograms
  void bookWheelHistos(int wheel, std::string histoTag, std::string folder="");

  /// checks for RPC Triggers
  bool hasRPCTriggers(const edm::Event& e);

  /// return the top folder
  std::string topFolder(std::string source) { return source=="DCC" ? "DT/03-LocalTrigger-DCC/" : "DT/04-LocalTrigger-DDU/"; }

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
  
  /// EndJob
  void endJob(void);
    
 private:
  
  int nevents;

  std::string SegmArbitration;

  bool processDCC, processDDU, detailedPlots;
  std::vector<std::string> processTags;
  int minBXDDU, maxBXDDU;

  float phiAccRange;
  int nMinHitsPhi;

  edm::InputTag inputTagMuons;

  edm::InputTag inputTagDCC;
  edm::InputTag inputTagDDU;
  edm::InputTag inputTagSEG;

  edm::InputTag inputTagGMT;

  DQMStore* dbe;
  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  DTTrigGeomUtils* trigGeomUtils;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;
  std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;

};

#endif
