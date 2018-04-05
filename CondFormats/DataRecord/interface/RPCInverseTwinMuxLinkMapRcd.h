#ifndef CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"

class RPCInverseTwinMuxLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseTwinMuxLinkMapRcd
                                                            , boost::mpl::vector<RPCTwinMuxLinkMapRcd> >
{};

#endif // CondFormats_DataRecord_RPCInverseTwinMuxLinkMapRcd_h
