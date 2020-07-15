#ifndef CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCCPPFLinkMapRcd.h"

class RPCInverseCPPFLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseCPPFLinkMapRcd,
                                                            boost::mp11::mp_list<RPCCPPFLinkMapRcd> > {};

#endif  // CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h
