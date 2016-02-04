#ifndef IOPool_Streamer_StreamThingProducer_h
#define IOPool_Streamer_StreamThingProducer_h

/** \class ThingProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include <string>
#include <vector>

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
    int inst_count_;
    std::vector<std::string> names_;
    int start_count_;

    bool apply_bit_mask_;
    unsigned int bit_mask_;
  };
}
#endif
