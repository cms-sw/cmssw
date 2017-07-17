#ifndef StoreESCondition_h
#define StoreESCondition_h

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <typeinfo>
#include <sstream>

#include "CondFormats/ESObjects/interface/ESTimeSampleWeights.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/ESObjects/interface/ESThresholds.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESMissingEnergyCalibration.h"
#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class  StoreESCondition : public edm::EDAnalyzer {

 public:

  ESThresholds* readESThresholdsFromFile(const char*);  
  ESPedestals* readESPedestalsFromFile(const char*);  
  ESRecHitRatioCuts* readESRecHitRatioCutsFromFile(const char*);  
  ESGain* readESGainFromFile(const char*);
  ESTimeSampleWeights* readESTimeSampleWeightsFromFile(const char*);
  ESChannelStatus* readESChannelStatusFromFile(const char *);
  ESIntercalibConstants* readESIntercalibConstantsFromFile(const char*);
  ESMissingEnergyCalibration* readESMissingEnergyFromFile(const char*);
  ESEEIntercalibConstants* readESEEIntercalibConstantsFromFile(const char*);
  void writeToLogFile(std::string , std::string, unsigned long long) ;
  void writeToLogFileResults(char* ) ;
  
  explicit  StoreESCondition(const edm::ParameterSet& iConfig );
  ~StoreESCondition();
  
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();
  
 private:
  
  void fillHeader(char*);
  
  std::vector< std::string > objectName_ ;
  std::vector< std::string > inpFileName_ ;
  std::vector< std::string > inpFileNameEE_ ;
  std::string prog_name_ ;
  std::vector< unsigned long long > since_; // beginning IOV for objects
  std::string logfile_;

  unsigned int esgain_;
  
  std::string to_string( char value[]) {
    std::ostringstream streamOut;
    streamOut << value;
    return streamOut.str();
  }
  
};

#endif
