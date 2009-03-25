#include <vector>

#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalRawData/interface/EcalListOfFEDS.h>
#include <DataFormats/EcalRawData/interface/ESListOfFEDS.h>

#include <DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h>
#include <DataFormats/EcalRawData/interface/ESKCHIPBlock.h>

#include <DataFormats/Common/interface/Wrapper.h>

namespace {
  struct dictionary {
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

    ESListOfFEDSCollection ESFedscol_ ;
    edm::Wrapper<ESListOfFEDSCollection> theESFedscol_ ;
    ESListOfFEDS t_ESListOfFEDS;
    edm::Wrapper<ESListOfFEDS> the_ESListOfFEDS;

    ESDCCHeaderBlock ESDCC_;
    std::vector<ESDCCHeaderBlock> vESDCC_;
    edm::SortedCollection<ESDCCHeaderBlock> scESDCC_;
    ESRawDataCollection ESDC_;
    edm::Wrapper<ESRawDataCollection> theESDC_;
    edm::Wrapper< edm::SortedCollection<ESDCCHeaderBlock> > theESDCC_;

    ESKCHIPBlock ESKCHIP_;
    std::vector<ESKCHIPBlock> vESKCHIP_;
    edm::SortedCollection<ESKCHIPBlock> scESKCHIP_;
    ESLocalRawDataCollection ESLDC_;
    edm::Wrapper<ESLocalRawDataCollection> theESLDC_;
    edm::Wrapper< edm::SortedCollection<ESKCHIPBlock> > theESKCHIP_;
  };
}
