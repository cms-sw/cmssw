#ifndef GEOMETRY_RECORDS_MUON_NUMBERING_RCD_H
#define GEOMETRY_RECORDS_MUON_NUMBERING_RCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include <boost/mp11/list.hpp>

class MuonNumberingRcd
    : public edm::eventsetup::DependentRecordImplementation<MuonNumberingRcd,
                                                            boost::mp11::mp_list<DDSpecParRegistryRcd>> {};
#endif
