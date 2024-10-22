#ifndef MuonNumberingInitialization_MuonNumberingRecord_h
#define MuonNumberingInitialization_MuonNumberingRecord_h
// -*- C++ -*-
//
// Package:     MuonNumberingInitialization
// Class  :     MuonNumberingRecord
//
/**\class MuonNumberingRecord MuonNumberingRecord.h Geometry/MuonNumberingInitialization/interface/MuonNumberingRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:
// Created:     Thu Sep 28 16:41:02 PDT 2006
//

#include <FWCore/Utilities/interface/mplVector.h>
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"

class MuonNumberingRecord
    : public edm::eventsetup::DependentRecordImplementation<MuonNumberingRecord,
                                                            edm::mpl::Vector<IdealGeometryRecord,
                                                                             CSCRecoDigiParametersRcd,
                                                                             CSCRecoGeometryRcd,
                                                                             DTRecoGeometryRcd,
                                                                             DDSpecParRegistryRcd,
                                                                             GeometryFileRcd> > {};

#endif
