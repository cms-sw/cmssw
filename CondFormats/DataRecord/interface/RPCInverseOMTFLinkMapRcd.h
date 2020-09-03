#ifndef CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h
#define CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"

class RPCInverseOMTFLinkMapRcd
    : public edm::eventsetup::DependentRecordImplementation<RPCInverseOMTFLinkMapRcd,
                                                            edm::mpl::Vector<RPCOMTFLinkMapRcd> > {};

#endif  // CondFormats_DataRecord_RPCInverseOMTFLinkMapRcd_h
