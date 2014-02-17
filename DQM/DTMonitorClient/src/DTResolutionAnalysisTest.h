#ifndef DTResolutionAnalysisTest_H
#define DTResolutionAnalysisTest_H


/** \class DTResolutionAnalysisTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2011/10/31 17:22:23 $
 *  $Revision: 1.12 $
 *  \author  G. Mila - INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <string>
#include <map>

class DTGeometry;
class DTSuperLayerId;
class DQMStore;
class MonitorElement;


class DTResolutionAnalysisTest: public edm::EDAnalyzer {

public:

  /// Constructor
  DTResolutionAnalysisTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTResolutionAnalysisTest();

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// book the summary histograms
  void bookHistos();
  void bookHistos(int wh);
  void bookHistos(int wh, int sect);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  void endRun(edm::Run const& run, edm::EventSetup const& c);



private:
  void resetMEs();

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  int percentual;

  DQMStore* dbe;

  // permitted test ranges
  double  maxGoodMeanValue;
  double  minBadMeanValue;
  double  maxGoodSigmaValue;
  double  minBadSigmaValue;

  bool doCalibAnalysis;

  edm::ESHandle<DTGeometry> muonGeom;

  // Histograms for tests
  std::map< std::pair<int,int> , MonitorElement* > MeanHistos;
  std::map< std::pair<int,int> , MonitorElement* > SigmaHistos;
  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelMeanHistos;
  std::map< int, MonitorElement* > wheelSigmaHistos;
  
  std::map< int, MonitorElement* > meanDistr;
  std::map< int, MonitorElement* > sigmaDistr;


  // Compute the station from the bin number of mean and sigma histos
  int stationFromBin(int bin) const;
  // Compute the sl from the bin number of mean and sigma histos
  int slFromBin(int bin) const;

  double meanInRange(double mean) const;
  double sigmaInRange(double sigma) const;

  MonitorElement* globalResSummary;
  
  // top folder for the histograms in DQMStore
  std::string topHistoFolder;

};

#endif
