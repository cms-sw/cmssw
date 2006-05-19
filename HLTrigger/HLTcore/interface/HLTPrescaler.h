#ifndef HLTPrescaler_H
#define HLTPrescaler_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm
{
  class HLTPrescaler : public edm::EDFilter
  {
  public:
    explicit HLTPrescaler(edm::ParameterSet const&);
    virtual ~HLTPrescaler();

    virtual bool filter(edm::Event& e, edm::EventSetup const& c);

  private:
    unsigned int n_;     // accept one in n_
    unsigned int count_;
  };
}

#endif

