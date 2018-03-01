#ifndef __L1Trigger_L1THGCal_HGCalTriggerTools_h__
#define __L1Trigger_L1THGCal_HGCalTriggerTools_h__

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Framework/interface/EventSetup.h"

class HGCalTriggerTools
{
  public:
    HGCalTriggerTools():
      eeLayers_(0), fhLayers_(0), bhLayers_(0), totalLayers_(0){}
    ~HGCalTriggerTools() {};

    void eventSetup(const edm::EventSetup&);
    unsigned layers(ForwardSubdetector type) const;
    unsigned layerWithOffset(unsigned) const;

  private:
    unsigned eeLayers_;
    unsigned fhLayers_;
    unsigned bhLayers_;
    unsigned totalLayers_;
};


#endif
