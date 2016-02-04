#ifndef SiStripDetVOffReader_H
#define SiStripDetVOffReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


//
//
// class decleration
//
class SiStripDetVOffReader : public edm::EDAnalyzer {

public:
  explicit SiStripDetVOffReader( const edm::ParameterSet& );
  ~SiStripDetVOffReader();
    
  void analyze( const edm::Event&, const edm::EventSetup& );
  
private:
  bool printdebug_;
  std::vector<uint32_t> detids;
};
#endif

/*  LocalWords:  ifndef
 */
