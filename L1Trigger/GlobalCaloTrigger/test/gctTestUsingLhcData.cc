#include "L1Trigger/GlobalCaloTrigger/test/gctTestUsingLhcData.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include <iostream>

gctTestUsingLhcData::gctTestUsingLhcData() { }
gctTestUsingLhcData::~gctTestUsingLhcData() { }

// Read the region Et values for a single event from a text file and prepare them to be loaded into the GCT
std::vector<L1CaloRegion> gctTestUsingLhcData::loadEvent(const edm::Event& iEvent, const int16_t bx)
{
  std::vector<L1CaloRegion> result;

  edm::InputTag inputDataTag("l1GctHwDigis");

  edm::Handle<std::vector<L1CaloRegion> > inputRegions;
  iEvent.getByLabel(inputDataTag, inputRegions);

  for (std::vector<L1CaloRegion>::const_iterator reg=inputRegions->begin();
       reg!=inputRegions->end(); reg++) {
    if (reg->bx() == bx) result.push_back(*reg);
  }

  return result;
}
