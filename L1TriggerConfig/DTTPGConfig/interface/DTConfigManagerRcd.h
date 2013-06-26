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
//		SV September 2008: create dependent record from DTCCBConfigRcd
//

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"

class DTT0Rcd;
class DTTPGParametersRcd;

class DTConfigManagerRcd : public
edm::eventsetup::DependentRecordImplementation<DTConfigManagerRcd,boost::mpl::vector<DTCCBConfigRcd,DTKeyedConfigListRcd,DTT0Rcd,DTTPGParametersRcd> > {};

#endif
