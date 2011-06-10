#ifndef DTNoiseAnalysisTest_H
#define DTNoiseAnalysisTest_H


/** \class DTNoiseAnalysisTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.5 $
 *  \author  G. Mila - INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>


#include <iostream>
#include <string>
#include <map>




class DTGeometry;
class DTChamberId;
class DTSuperLayerId;
class DQMStore;
class MonitorElement;

class DTNoiseAnalysisTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTNoiseAnalysisTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTNoiseAnalysisTest();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context) ;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// book the summary histograms
  void bookHistos();

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);


private:

  /// Get the ME name
  std::string getMEName(const DTChamberId & chID);
  std::string getSynchNoiseMEName(int wheelId) const;


  int nevents;
  int nMinEvts;
  
  DQMStore* dbe;
  
  // the dt geometry
  edm::ESHandle<DTGeometry> muonGeom;

  // paramaters from cfg
  int noisyCellDef;

  // wheel summary histograms  
  std::map< int, MonitorElement* > noiseHistos;
  std::map< int, MonitorElement* > noisyCellHistos;
  MonitorElement* summaryNoiseHisto;
  MonitorElement* summarySynchNoiseHisto;
  MonitorElement* glbSummarySynchNoiseHisto;

  bool doSynchNoise;
  double maxSynchNoiseRate;
};

#endif
