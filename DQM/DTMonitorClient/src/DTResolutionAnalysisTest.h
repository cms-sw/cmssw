#ifndef DTResolutionAnalysisTest_H
#define DTResolutionAnalysisTest_H


/** \class DTResolutionAnalysisTest
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
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <string>
#include <map>

class DTGeometry;
class DTSuperLayerId;
class DQMStore;
class MonitorElement;

class DTResolutionAnalysisTest: public DQMEDHarvester {

public:

  /// Constructor
  DTResolutionAnalysisTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTResolutionAnalysisTest();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  void bookHistos(DQMStore::IBooker &);
  void bookHistos(DQMStore::IBooker &,int wh);
  void bookHistos(DQMStore::IBooker &,int wh, int sect);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

private:
  void resetMEs();

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  int percentual;

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
