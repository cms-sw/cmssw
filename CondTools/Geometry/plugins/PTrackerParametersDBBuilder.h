#ifndef CondToolsGeometry_PTrackerParametersDBBuilder_h
#define CondToolsGeometry_PTrackerParametersDBBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PTrackerParametersDBBuilder : public edm::EDAnalyzer
{
 public:
  explicit PTrackerParametersDBBuilder( const edm::ParameterSet& ) {}
  ~PTrackerParametersDBBuilder( void ) {}
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {}
};

#endif
