#ifndef DTResolutionAnalysisTest_H
#define DTResolutionAnalysisTest_H


/** \class DTResolutionAnalysisTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/12/02 13:29:11 $
 *  $Revision: 1.4 $
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
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// book the summary histograms
  void bookHistos(int wh);
  void bookHistos(int wh, int sect);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);



private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  int percentual;
  std::string folderRoot;

  DQMStore* dbe;

  // permitted test ranges
  double permittedMeanRange; 
  double permittedSigmaRange; 

  edm::ESHandle<DTGeometry> muonGeom;

  // Histograms for tests
  std::map< std::pair<int,int> , MonitorElement* > MeanHistos;
  std::map< std::pair<int,int> , MonitorElement* > SigmaHistos;
  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelMeanHistos;
  std::map< int, MonitorElement* > wheelSigmaHistos;
 
  // Compute the station from the bin number of mean and sigma histos
  int stationFromBin(int bin) const;
  // Compute the sl from the bin number of mean and sigma histos
  int slFromBin(int bin) const;

};

#endif
