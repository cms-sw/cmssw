#ifndef SiStripClusterInfoProducer_h
#define SiStripClusterInfoProducer_h

#include <iostream> 
#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"

class Event;
class EventSetup;
class SiStripPedestalsService;
class SiStripNoiseService;
class SiStripRawDigi;
class SiStripCluster;
class SiStripClusterInfo;
class SiStripPedestalsSubtractor;
class SiStripCommonModeNoiseSubtractor;

namespace cms
{
  class SiStripClusterInfoProducer : public edm::EDProducer
  {
  public:

    explicit SiStripClusterInfoProducer(const edm::ParameterSet& conf);

    virtual ~SiStripClusterInfoProducer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

    void cluster_algorithm(const edm::DetSetVector<SiStripCluster>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output);
    void digi_algorithm(const edm::DetSetVector<SiStripRawDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output,std::string rawdigiLabel);
  private:
    edm::ParameterSet conf_;
    SiStripNoiseService SiStripNoiseService_;  
    bool RawModeRun_;
    uint16_t _NEIGH_STRIP_;

    SiStripCommonModeNoiseSubtractor* SiStripCommonModeNoiseSubtractor_;
    std::string CMNSubtractionMode_;
    bool validCMNSubtraction_;  

    SiStripPedestalsSubtractor* SiStripPedestalsSubtractor_;
  };
}
#endif
