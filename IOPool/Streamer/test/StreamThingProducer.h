#ifndef EDMREFTEST_THINGPRODUCER_H
#define EDMREFTEST_THINGPRODUCER_H

/** \class ThingProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest_thing
{
  class StreamThingProducer : public edm::EDProducer
  {
  public:
    explicit StreamThingProducer(edm::ParameterSet const& ps);

    virtual ~StreamThingProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    int size_;
  };
}
#endif
