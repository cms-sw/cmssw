#ifndef CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCCPPFLinkMapRcd.h"

class RPCInverseCPPFLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseCPPFLinkMapRcd,
                                                            edm::mpl::Vector<RPCCPPFLinkMapRcd> > {};

#endif  // CondFormats_DataRecord_RPCInverseCPPFLinkMapRcd_h
