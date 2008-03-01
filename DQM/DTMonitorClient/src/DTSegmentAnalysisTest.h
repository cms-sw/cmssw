#ifndef DTSegmentAnalysisTest_H
#define DTSegmentAnalysisTest_H


/** \class DTSegmentAnalysisTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/11/29 13:34:30 $
 *  $Revision: 1.1 $
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

class DTSegmentAnalysisTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTSegmentAnalysisTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTSegmentAnalysisTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// book the summary histograms
  void bookHistos(int wh);

  /// Get the ME name
  std::string getMEName(const DTChamberId & chID, std::string histoTag);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);


private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  int badChpercentual;

  DQMStore* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelHistos;

};

#endif

