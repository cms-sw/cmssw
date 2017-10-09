#ifndef DTNoiseAnalysisTest_H
#define DTNoiseAnalysisTest_H


/** \class DTNoiseAnalysisTest
 * *
 *  DQM Test Client
 *
 *  \author  G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <DQMServices/Core/interface/DQMEDHarvester.h>


#include <iostream>
#include <string>
#include <map>




class DTGeometry;
class DTChamberId;
class DTSuperLayerId;
class DQMStore;
class MonitorElement;

class DTNoiseAnalysisTest: public DQMEDHarvester{

public:

  /// Constructor
  DTNoiseAnalysisTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTNoiseAnalysisTest();

protected:

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context) ;

  /// book the summary histograms

  void bookHistos(DQMStore::IBooker &);

  /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

private:

  /// Get the ME name
  std::string getMEName(const DTChamberId & chID);
  std::string getSynchNoiseMEName(int wheelId) const;


  int nevents;
  int nMinEvts;
  
  bool bookingdone;
  
  // the dt geometry
  edm::ESHandle<DTGeometry> muonGeom;

  // paramaters from cfg
  int noisyCellDef;

  // wheel summary histograms  
  std::map< int, MonitorElement* > noiseHistos;
  std::map< int, MonitorElement* > noisyCellHistos;
  MonitorElement* summaryNoiseHisto;
  MonitorElement* threshChannelsHisto;
  MonitorElement* summarySynchNoiseHisto;
  MonitorElement* glbSummarySynchNoiseHisto;

  bool doSynchNoise;
  bool detailedAnalysis;
  double maxSynchNoiseRate;
};

#endif
