#ifndef CondFormats_L1RPCConeBuilderRcd_h
#define CondFormats_L1RPCConeBuilderRcd_h
// -*- C++ -*-
//
// Package:     CondFormats
// Class  :     L1RPCConeBuilderRcd
// 
/**\class L1RPCConeBuilderRcd L1RPCConeBuilderRcd.h src/CondFormats/interface/L1RPCConeBuilderRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Fri Feb 22 12:15:57 CET 2008
// $Id: L1RPCConeBuilderRcd.h,v 1.2 2008/03/03 07:09:47 wsun Exp $
//

// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
// class L1RPCConeBuilderRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RPCConeBuilderRcd> {};

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"
#include <boost/mpl/vector.hpp>
class L1RPCConeBuilderRcd : public edm::eventsetup::DependentRecordImplementation<L1RPCConeBuilderRcd, boost::mpl::vector<MuonGeometryRecord,L1TriggerKeyListRcd,L1TriggerKeyRcd, L1RPCConeDefinitionRcd > > {};


#endif
