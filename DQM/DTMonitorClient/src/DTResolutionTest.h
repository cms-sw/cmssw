#ifndef DTResolutionTest_H
#define DTResolutionTest_H


/** \class DTResolutionTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.13 $
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

class DTResolutionTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTResolutionTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTResolutionTest();

protected:

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookHistos(const DTChamberId & ch);

  /// book the summary histograms
  void bookHistos(int wh);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);
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

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  // histograms: < detRawID, Histogram >
  std::map< std::pair<int,int> , MonitorElement* > MeanHistos;
  std::map< std::pair<int,int> , MonitorElement* > SigmaHistos;
  std::map< std::pair<int,int> , MonitorElement* > SlopeHistos;
  std::map< std::string , MonitorElement* > MeanHistosSetRange;
  std::map< std::string , MonitorElement* > SigmaHistosSetRange;
  std::map< std::string , MonitorElement* > SlopeHistosSetRange;
  std::map< std::string , MonitorElement* > MeanHistosSetRange2D;
  std::map< std::string , MonitorElement* > SigmaHistosSetRange2D;
  std::map< std::string , MonitorElement* > SlopeHistosSetRange2D;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelMeanHistos;
  std::map< int, MonitorElement* > wheelSigmaHistos;
  std::map< int, MonitorElement* > wheelSlopeHistos;

  // cms summary histograms
  std::map <std::pair<int,int>, int> cmsMeanHistos;
  std::map <std::pair<int,int>, bool> MeanFilled;
  std::map <std::pair<int,int>, int> cmsSigmaHistos;
  std::map <std::pair<int,int>, bool> SigmaFilled;
  std::map <std::pair<int,int>, int> cmsSlopeHistos;
  std::map <std::pair<int,int>, bool> SlopeFilled;

  // Compute the station from the bin number of mean and sigma histos
  int stationFromBin(int bin) const;
  // Compute the sl from the bin number of mean and sigma histos
  int slFromBin(int bin) const;


};

#endif
