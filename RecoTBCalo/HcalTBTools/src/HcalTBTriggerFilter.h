#ifndef HCALTBTRIGGERFILTER_H
#define HCALTBTRIGGERFILTER_H 1

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

/** \class HcalTBTriggerFilter
    
   \author J. Mans - Minnesota
*/
class HcalTBTriggerFilter : public edm::global::EDFilter<> {
public:
  HcalTBTriggerFilter(const edm::ParameterSet& ps);
  bool filter(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

private:
  bool allowPedestal_;
  bool allowPedestalInSpill_;
  bool allowPedestalOutSpill_;
  bool allowLaser_;
  bool allowLED_;
  bool allowBeam_;
  edm::EDGetTokenT<HcalTBTriggerData> tok_tb_;
};

#endif
