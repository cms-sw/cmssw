#ifndef DTNoiseTest_H
#define DTNoiseTest_H



/** \class DTNoiseTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.8 $
 *  A. Gresele - INFN Trento
 *  G. Mila - INFN Torino
 *  M. Zanetti - CERN PH
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

class DTNoiseTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTNoiseTest(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTNoiseTest();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookHistos(const DTChamberId & ch, std::string folder, std::string histoTag);
  void bookHistos(const DTLayerId & ch, int nWire,std::string folder, std::string histoTag);

  /// Get the ME name
  std::string getMEName(const DTChamberId & ch);
  std::string getMEName(const DTLayerId & ly);



  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

 


private:

  bool debug;
  int updates;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;

  DQMStore* dbe;
  
  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  // the collection of noisy channels
  //std::map< uint32_t, std::vector<DTWireId> > theNoisyChannels;
   
  std::vector<DTWireId>  theNoisyChannels;

  // histograms: < detRawID, Histogram >
  //std::map<  uint32_t , MonitorElement* > histos;
  std::map<std::string, std::map<uint32_t, MonitorElement*> > histos;

};

#endif
