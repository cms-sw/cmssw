#ifndef CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"

class RPCInverseLBLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseLBLinkMapRcd, edm::mpl::Vector<RPCLBLinkMapRcd> > {
};

#endif  // CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h
