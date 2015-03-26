#ifndef DTResolutionTest_H
#define DTResolutionTest_H


/** \class DTResolutionTest
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
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTSuperLayerId;

class DTResolutionTest: public DQMEDHarvester{

public:

  /// Constructor
  DTResolutionTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTResolutionTest();

protected:

  /// Endjob
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  /// book the new ME
  void bookHistos(DQMStore::IBooker &, const DTChamberId & ch);

  /// book the summary histograms
  void bookHistos(DQMStore::IBooker &, int wh);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);
  std::string getMEName2D(const DTSuperLayerId & slID);

  /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);


private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  int percentual;

  bool bookingdone;

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
