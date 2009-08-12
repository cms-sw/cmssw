// Purpose: filter only needed events for LAS
// Original Author:  Adrian Perieanu
//         Created:  11th of August

// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class declaration
class LaserAlignmentEventFilter : public edm::EDFilter {

public:
  explicit LaserAlignmentEventFilter( const edm::ParameterSet& );
  ~LaserAlignmentEventFilter();
  
private:
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual bool filter( edm::Event&, const edm::EventSetup& );
  virtual void endJob();

  // container for cfg data
  int runFirst;
  int runLast;
  int eventFirst;
  int eventLast;
};

