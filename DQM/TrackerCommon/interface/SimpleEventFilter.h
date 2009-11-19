#ifndef SimpleEventFilter_H
#define SimpleEventFilter_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class SimpleEventFilter : public edm::EDFilter {
 public:
 SimpleEventFilter( const edm::ParameterSet & );
 ~SimpleEventFilter();
  private:
  bool filter( edm::Event &, edm::EventSetup const& );
  int nEvent_;
  int nInterval_;
  bool verbose_;
};

#endif
