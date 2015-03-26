#ifndef DTtTrigCalibrationTest_H
#define DTtTrigCalibrationTest_H

/** \class DTtTrigCalibrationTest
 * *
 *  DQM Test Client
 *
 *  \author  M. Zanetti CERN
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
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
class DTTtrig;
class DTTimeBoxFitter;


class DTtTrigCalibrationTest: public DQMEDHarvester {

public:

  /// Constructor
  DTtTrigCalibrationTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTtTrigCalibrationTest();

protected:

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

  /// book the new ME

  void bookHistos(DQMStore::IBooker &,const DTChamberId & ch);

  /// book the summary histograms

  void bookHistos(DQMStore::IBooker &,const DTChamberId & ch, int wh);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);

  /// DQM Client Diagnostic

  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);


private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  int percentual;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  DTTimeBoxFitter *theFitter;

  // histograms: < detRawID, Histogram >
  std::map<  uint32_t , MonitorElement* > histos;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelHistos;

};

#endif
