#ifndef ModulesPrescaler_H
#define ModulesPrescaler_H

#include "FWCore/Framework/interface/EDFilter.h"

namespace edm
{
  class Prescaler : public edm::EDFilter
  {
  public:
    explicit Prescaler(edm::ParameterSet const&);
    virtual ~Prescaler();

    virtual bool filter(edm::Event& e, edm::EventSetup const& c);
    void endJob();

  private:
    int count_;
    int n_; // accept one in n
  };
}

#endif

