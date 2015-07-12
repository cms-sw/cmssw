#ifndef DTChamberEfficiencyTest_H
#define DTChamberEfficiencyTest_H


/** \class DTChamberEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  \author  G. Mila - INFN Torino
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
class DTSuperLayerId;
class DTLayerId;

class DTChamberEfficiencyTest: public DQMEDHarvester{

public:

  /// Constructor
  DTChamberEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTChamberEfficiencyTest();

protected:

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  /// book the new ME
  void bookHistos(DQMStore::IBooker &, const DTChamberId & ch);

  /// book the report summary
  void bookHistos(DQMStore::IBooker &);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTChamberId & chID);

  /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);



private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;

  bool bookingdone;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  std::map< std::string , MonitorElement* > xEfficiencyHistos;
  std::map< std::string , MonitorElement* > yEfficiencyHistos;
  std::map< std::string , MonitorElement* > xVSyEffHistos;
  std::map< int, MonitorElement* > summaryHistos;

};

#endif
