#ifndef DTTriggerEfficiencyTask_H
#define DTTriggerEfficiencyTask_H

/*
 * \file DTTriggerEfficiencyTask.h
 *
 * $Date: 2009/06/15 14:59:22 $
 * $Revision: 1.0 $
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
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DTRecSegment4D;
class DTLocalTrigger;
class DTTrigGeomUtils;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;

class DTTriggerEfficiencyTask: public edm::EDAnalyzer{
  
 public:
  
  /// Constructor
  DTTriggerEfficiencyTask(const edm::ParameterSet& ps );
  
  /// Destructor
  virtual ~DTTriggerEfficiencyTask();
  
 protected:
  
  // BeginJob
  void beginJob(const edm::EventSetup& c);
  
  /// Book chamber granularity histograms
  void bookChamberHistos(const DTChamberId& dtCh, std::string histoTag, std::string folder="");

  /// Book wheel granularity histograms
  void bookWheelHistos(int wheel, std::string histoTag, std::string folder="");

  /// checks for RPC Triggers
  bool hasRPCTriggers(const edm::Event& e);

  /// return the top folder 0=DDU 1=DCC
  std::string topFolder(bool source) { return source ? "DT/03-LocalTrigger-DCC/" : "DT/04-LocalTrigger-DDU/"; }

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
  
  /// EndJob
  void endJob(void);
    
 private:
  
  int nevents;

  bool processDCC, processDDU, detailedPlots;
  int minBXDDU, maxBXDDU;

  int phCodeBestDCC[6][5][13];
  int phCodeBestDDU[6][5][13];
  const L1MuDTChambPhDigi* phBestDCC[6][5][13];
  const DTLocalTrigger*    phBestDDU[6][5][13];

  DQMStore* dbe;
  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  DTTrigGeomUtils* trigGeomUtils;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;
  std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;

};

#endif
