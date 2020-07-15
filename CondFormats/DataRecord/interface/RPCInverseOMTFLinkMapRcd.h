#ifndef CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"

class RPCInverseOMTFLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseOMTFLinkMapRcd,
                                                            boost::mp11::mp_list<RPCOMTFLinkMapRcd> > {};

#endif  // CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h
