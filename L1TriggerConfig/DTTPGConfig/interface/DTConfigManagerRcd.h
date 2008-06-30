#ifndef DTTPGConfig_DTConfigManagerRcd_h
#define DTTPGConfig_DTConfigManagerRcd_h
// -*- C++ -*-
//
// Package:     DTTPGConfig
// Class  :     DTConfigRcd
// 
/**\class  DTConfigRcd DTConfigRcd.h L1TriggerConfig/DTTPGConfig/interface/DTConfigRcd.h

 Description: Record for storing TPG chip configurations in Event Setup

 Usage:
    <usage>

*/
//
// Author:      Sara Vanini
// Created:     Mar  30 16:49:43 CEST 2007
// $Id: 
//
#include "boost/mpl/vector.hpp"
//#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

//class DTConfigManagerRcd : public edm::eventsetup::DependentRecordImplementation<DTConfigManagerRcd, boost::mpl::vector<MuonGeometryRecord> > {};
class DTConfigManagerRcd : public edm::eventsetup::EventSetupRecordImplementation<DTConfigManagerRcd> {};

#endif
