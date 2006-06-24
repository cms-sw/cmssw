#ifndef HLTPrescaler_H
#define HLTPrescaler_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HLTPrescaler : public edm::EDFilter {

#include "HLTrigger/HLTcore/interface/HLTadd.h"

 public:

  explicit HLTPrescaler(edm::ParameterSet const&);
  virtual ~HLTPrescaler();

  virtual bool filter(edm::Event& e, edm::EventSetup const& c);

  private:
    bool         b_;     // to put a filterobject into the event?
    unsigned int n_;     // accept one in n_
    unsigned int count_; // local event counter

};

#endif

