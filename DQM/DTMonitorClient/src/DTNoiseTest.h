#ifndef DTNoiseTest_H
#define DTNoiseTest_H



/** \class DTNoiseTest
 * *
 *  DQM Test Client
 *
 *  A. Gresele - INFN Trento
 *  G. Mila - INFN Torino
 *  M. Zanetti - CERN PH
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include <CondFormats/DataRecord/interface/DTStatusFlagRcd.h>
#include <CondFormats/DTObjects/interface/DTStatusFlag.h>


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTSuperLayerId;
class DTLayerId ;
class DTWireId;

class DTNoiseTest: public DQMEDHarvester{

public:

  /// Constructor
  DTNoiseTest(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTNoiseTest();

protected:

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  /// book the new ME
  void bookHistos(DQMStore::IBooker &, const DTChamberId & ch, std::string folder, std::string histoTag);
  void bookHistos(DQMStore::IBooker &, const DTLayerId & ch, int nWire,std::string folder, std::string histoTag);

  /// Get the ME name
  std::string getMEName(const DTChamberId & ch);
  std::string getMEName(const DTLayerId & ly);

  /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);
 


private:

  bool debug;
  int updates;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;

  bool bookingdone;
  
  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  // the collection of noisy channels   
  std::vector<DTWireId>  theNoisyChannels;

  // histograms: < detRawID, Histogram >
  std::map<std::string, std::map<uint32_t, MonitorElement*> > histos;

};

#endif
