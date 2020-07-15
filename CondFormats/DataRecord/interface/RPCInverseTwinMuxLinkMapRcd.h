#ifndef CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"

class RPCInverseTwinMuxLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseTwinMuxLinkMapRcd,
                                                            boost::mp11::mp_list<RPCTwinMuxLinkMapRcd> > {};

#endif  // CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h
