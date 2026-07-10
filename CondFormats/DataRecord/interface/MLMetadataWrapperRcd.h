#ifndef DataRecord_MLMetadataWrapperRcd_h
#define DataRecord_MLMetadataWrapperRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/MLMetadataRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class MLMetadataWrapperRcd
    : public edm::eventsetup::DependentRecordImplementation<MLMetadataWrapperRcd, edm::mpl::Vector<MLMetadataRcd> > {};

#endif
