#ifndef SiStripClusterInfoProducer_h
#define SiStripClusterInfoProducer_h

#include <iostream> 
#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"

class Event;
class EventSetup;
class SiStripNoiseService;
class SiStripCluster;
class SiStripClusterInfo;

namespace cms
{
  class SiStripClusterInfoProducer : public edm::EDProducer
  {
  public:

    explicit SiStripClusterInfoProducer(const edm::ParameterSet& conf);

    virtual ~SiStripClusterInfoProducer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

    void algorithm(const edm::DetSetVector<SiStripCluster>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output);
  private:
    edm::ParameterSet conf_;
    SiStripNoiseService SiStripNoiseService_;  
  };
}
#endif
