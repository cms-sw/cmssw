#ifndef CondToolsGeometry_PHcalParametersDBBuilder_h
#define CondToolsGeometry_PHcalParametersDBBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PHcalParametersDBBuilder : public edm::EDAnalyzer
{
 public:
  explicit PHcalParametersDBBuilder( const edm::ParameterSet& ) {}
  ~PHcalParametersDBBuilder( void ) {}
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {}
};

#endif
