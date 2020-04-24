#ifndef CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCCPPFLinkMapRcd.h"

class RPCInverseCPPFLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseCPPFLinkMapRcd
                                                            , boost::mpl::vector<RPCCPPFLinkMapRcd> >
{};

#endif // CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h
