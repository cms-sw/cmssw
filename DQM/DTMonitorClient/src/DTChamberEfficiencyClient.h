#ifndef DTChamberEfficiencyClient_H
#define DTChamberEfficiencyClient_H


/** \class DTChamberEfficiencyClient
 * *
 *  DQM Test Client
 *
 *  \author  M. Pelliccioni - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTLayerId;

class DTChamberEfficiencyClient: public DQMEDHarvester{

public:

  /// Constructor
  DTChamberEfficiencyClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTChamberEfficiencyClient();

protected:

  void beginRun(const edm::Run& , const edm::EventSetup&);
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  /// book the report summary

  void bookHistos(DQMStore::IBooker &);
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);

private:

  unsigned int nLumiSegs;
  int prescaleFactor;

  edm::ESHandle<DTGeometry> muonGeom;

  //an histogram of efficiency for each wheel, for each quality scenario
  MonitorElement* summaryHistos[5][2];
  MonitorElement* globalEffSummary;
 
  MonitorElement* globalEffDistr;
  std::map<int, MonitorElement*> EffDistrPerWh;

};

#endif
