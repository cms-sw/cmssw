#ifndef DTDeadChannelTest_H
#define DTDeadChannelTest_H


/** \class DTDeadChannelTest
 * *
 *  DQM Test Client
 *
 *  \author  G. Mila - INFN Torino
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
class DTTtrig;

class DTDeadChannelTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDeadChannelTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTDeadChannelTest();

protected:

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// book the new ME
  void bookHistos(DQMStore::IBooker &, const DTLayerId & ch, int firstWire, int lastWire);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTChamberId & chId);

  
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);

private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  std::map< std::string , MonitorElement* > OccupancyDiffHistos;
  
};

#endif
