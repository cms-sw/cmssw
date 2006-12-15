#ifndef SiStripClusterInfoProducer_h
#define SiStripClusterInfoProducer_h

#include <iostream> 
#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsService.h"

class Event;
class EventSetup;
class SiStripNoiseService;
//class SiStripPedestalsService;
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
    void digi_algorithm(const edm::DetSetVector<SiStripDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output);
    void rawdigi_algorithm(const edm::DetSetVector<SiStripRawDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output,std::string rawdigiLabel);
    void findNeigh(std::vector< edm::DetSet<SiStripClusterInfo> >::iterator output_iter,std::vector<int16_t>& vstrip,std::vector<int16_t>& vadc,char* mode);
  
  private:
    edm::ParameterSet conf_;
    SiStripNoiseService SiStripNoiseService_;  
    SiStripPedestalsService SiStripPedestalsService_; 
    uint16_t _NEIGH_STRIP_;

    SiStripCommonModeNoiseSubtractor* SiStripCommonModeNoiseSubtractor_;
    std::string CMNSubtractionMode_;
    bool validCMNSubtraction_;  

    SiStripPedestalsSubtractor* SiStripPedestalsSubtractor_;
  };
}
#endif
