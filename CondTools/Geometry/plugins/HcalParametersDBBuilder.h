#ifndef CondToolsGeometry_HcalParametersDBBuilder_h
#define CondToolsGeometry_HcalParametersDBBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalParametersDBBuilder : public edm::EDAnalyzer {

public:
  explicit HcalParametersDBBuilder( const edm::ParameterSet& ) {}
  ~HcalParametersDBBuilder( void ) {}
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {}
};

#endif
