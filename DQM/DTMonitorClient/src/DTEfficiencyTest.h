#ifndef DTEfficiencyTest_H
#define DTEfficiencyTest_H


/** \class DTEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.8 $
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

class DTEfficiencyTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTEfficiencyTest();

protected:

  /// BeginJob
  void beginJob();

  /// Analyze
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookHistos(const DTLayerId & ch, int firstWire, int lastWire);

  /// book the summary histograms
  void bookHistos(int wh);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTLayerId & lID);

  
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);



private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  int percentual;

  DQMStore* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  std::map< DTLayerId , MonitorElement* > EfficiencyHistos;
  std::map< DTLayerId , MonitorElement* > UnassEfficiencyHistos;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelHistos;  
  std::map< int, MonitorElement* > wheelUnassHistos;
  
};

#endif
