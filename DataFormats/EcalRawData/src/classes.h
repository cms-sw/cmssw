#include <vector>
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalRawData/interface/EcalListOfFEDS.h>
#include <DataFormats/Common/interface/Wrapper.h>

namespace {
  namespace {
    EcalDCCHeaderBlock ERDC_;
    EcalDCCHeaderBlock::EcalDCCEventSettings ERDCSet_;
    std::vector< EcalDCCHeaderBlock > vERDC_;
    edm::SortedCollection<EcalDCCHeaderBlock> scERDC_;
    EcalRawDataCollection theERDC_;
    edm::Wrapper<EcalRawDataCollection> anotherERDCw_;
    edm::Wrapper< edm::SortedCollection<EcalDCCHeaderBlock> > theEDHBw_;

    EcalListOfFEDSCollection Fedscol_ ;
    edm::Wrapper<EcalListOfFEDSCollection> theFedscol_ ;
    EcalListOfFEDS t_EcalListOfFEDS;
    edm::Wrapper<EcalListOfFEDS> the_EcalListOfFEDS;
  }
}
