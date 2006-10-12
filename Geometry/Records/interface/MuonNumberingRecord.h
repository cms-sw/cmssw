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
// $Id$
//

#include <boost/mpl/vector.hpp>
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"


class MuonNumberingRecord : public edm::eventsetup::DependentRecordImplementation<MuonNumberingRecord, boost::mpl::vector<IdealGeometryRecord> > {};

#endif
