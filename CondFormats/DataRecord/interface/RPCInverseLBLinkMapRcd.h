#ifndef CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"

class RPCInverseLBLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseLBLinkMapRcd,
                                                            boost::mp11::mp_list<RPCLBLinkMapRcd> > {};

#endif  // CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h
