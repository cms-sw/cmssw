#ifndef GEOMETRY_RECORDS_MUON_NUMBERING_RCD_H
#define GEOMETRY_RECORDS_MUON_NUMBERING_RCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class MuonNumberingRcd
    : public edm::eventsetup::DependentRecordImplementation<MuonNumberingRcd, edm::mpl::Vector<DDSpecParRegistryRcd>> {
};
#endif
