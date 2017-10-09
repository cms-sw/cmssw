#ifndef SiStripFedCablingBuilder_H
#define SiStripFedCablingBuilder_H
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class SiStripFedCablingBuilder : public edm::EDAnalyzer {

 public:

  SiStripFedCablingBuilder( const edm::ParameterSet& iConfig);

  ~SiStripFedCablingBuilder(){};

  void beginRun( const edm::Run&, const edm::EventSetup& );

  void analyze(const edm::Event& , const edm::EventSetup& ) {;}

 private:

  bool printFecCabling_;
  bool printDetCabling_;
  bool printRegionCabling_;

};
#endif
