#ifndef DTSegmentAnalysisTest_H
#define DTSegmentAnalysisTest_H


/** \class DTSegmentAnalysisTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2012/04/13 10:54:40 $
 *  $Revision: 1.11 $
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


  /// BeginJob
  void beginJob();
  void endJob(void);

  void beginRun(const edm::Run& run, const edm::EventSetup& eSetup);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// book the summary histograms
  void bookHistos();

  /// Get the ME name
  std::string getMEName(const DTChamberId & chID, std::string histoTag);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Perform client diagnostic operations
  void performClientDiagnostic();

  /// DQM Client Diagnostic in online mode
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

  /// DQM Client Diagnostic in offline mode
  void endRun(edm::Run const& run, edm::EventSetup const& c);


private:

  int nevents;
  unsigned int nLumiSegs;
  // switch on for detailed analysis
  bool detailedAnalysis;
  int nMinEvts;

  int maxPhiHit;
  int maxPhiZHit;

  bool runOnline;

  DQMStore* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  // the histograms  
  std::map< std::pair<int,int>, MonitorElement* > chi2Histos;
  std::map< std::pair<int,int>, MonitorElement* > segmRecHitHistos;
  std::map< int, MonitorElement* > summaryHistos;
  bool normalizeHistoPlots;
  // top folder for the histograms in DQMStore
  std::string topHistoFolder;
  // hlt DQM mode
  bool hltDQMMode;
};

#endif

