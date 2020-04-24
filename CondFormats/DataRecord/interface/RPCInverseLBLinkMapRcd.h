#ifndef CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"

class RPCInverseLBLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseLBLinkMapRcd
                                                            , boost::mpl::vector<RPCLBLinkMapRcd> >
{};

#endif // CondFormats_DataRecord_RPCInverseLBLinkMapRcd_h
