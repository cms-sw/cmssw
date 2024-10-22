#ifndef CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"

class RPCInverseTwinMuxLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseTwinMuxLinkMapRcd,
                                                            edm::mpl::Vector<RPCTwinMuxLinkMapRcd> > {};

#endif  // CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h
