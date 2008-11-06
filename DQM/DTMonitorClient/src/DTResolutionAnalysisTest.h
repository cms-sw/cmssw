#ifndef DTResolutionAnalysisTest_H
#define DTResolutionAnalysisTest_H


/** \class DTResolutionAnalysisTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/11/05 17:43:24 $
 *  $Revision: 1.1 $
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

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the summary histograms
  void bookHistos(int wh);

  /// Get the ME name
  std::string getMEName2D(const DTSuperLayerId & slID);

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

  // quality test names
  std::string MeanCriterionName; 
  std::string SigmaCriterionName; 

  edm::ESHandle<DTGeometry> muonGeom;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelMeanHistos;
  std::map< int, MonitorElement* > wheelSigmaHistos;
  std::map< int, MonitorElement* > wheelSlopeHistos;
 

  // Compute the station from the bin number of mean and sigma histos
  int stationFromBin(int bin) const;
  // Compute the sl from the bin number of mean and sigma histos
  int slFromBin(int bin) const;

};

#endif
