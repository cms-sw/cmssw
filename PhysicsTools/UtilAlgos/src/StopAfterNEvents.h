#ifndef RecoQA_StopAfterNEvents_h
#define RecoQA_StopAfterNEvents_h
// $Id: StopAfterNEvents.h,v 1.2 2006/04/23 16:11:08 wmtan Exp $
#include "FWCore/Framework/interface/EDFilter.h"

namespace edm {
  class ParameterSet;
}

class StopAfterNEvents : public edm::EDFilter {
public:
  StopAfterNEvents( const edm::ParameterSet & );
  ~StopAfterNEvents();
private:
  bool filter( edm::Event &, edm::EventSetup const& );
  const int nMax_;
  int n_;
  const bool verbose_;
};

#endif
