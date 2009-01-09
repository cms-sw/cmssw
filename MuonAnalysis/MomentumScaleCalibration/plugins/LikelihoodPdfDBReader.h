#ifndef LikelihoodPdfDBReader_H
#define LikelihoodPdfDBReader_H

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

class LikelihoodPdfDBReader : public edm::EDAnalyzer {

 public:
  explicit LikelihoodPdfDBReader( const edm::ParameterSet& );
  ~LikelihoodPdfDBReader();
  
  void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  //  uint32_t printdebug_;

};
#endif
