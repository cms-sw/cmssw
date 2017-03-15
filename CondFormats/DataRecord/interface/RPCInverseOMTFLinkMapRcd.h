#ifndef CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"

class RPCInverseOMTFLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseOMTFLinkMapRcd
                                                            , boost::mpl::vector<RPCOMTFLinkMapRcd> >
{};

#endif // CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h
