#ifndef SimpleEventFilter_H
#define SimpleEventFilter_H

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SimpleEventFilter : public edm::stream::EDFilter<> {
public:
  SimpleEventFilter(const edm::ParameterSet &);
  ~SimpleEventFilter() override;

private:
  bool filter(edm::Event &, edm::EventSetup const &) override;
  int nEvent_;
  int nInterval_;
  bool verbose_;
};

#endif
