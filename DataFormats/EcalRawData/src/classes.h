#include <vector>
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <FWCore/EDProduct/interface/Wrapper.h>

namespace {
  namespace {
    std::vector< EcalDCCHeaderBlock > vERDC_;
    edm::SortedCollection<EcalDCCHeaderBlock> scERDC_;
    EcalRawDataCollection theERDC_;
    edm::Wrapper<EcalRawDataCollection> anotherERDCw_;
    edm::Wrapper< edm::SortedCollection<EcalDCCHeaderBlock> > theEDHBw_;
  }
}
