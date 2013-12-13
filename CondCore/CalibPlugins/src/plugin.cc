/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/Calibration/interface/Conf.h"
#include "CondFormats/DataRecord/interface/PedestalsRcd.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/DataRecord/interface/mySiStripNoisesRcd.h"
#include "CondFormats/Calibration/interface/Efficiency.h"
#include "CondFormats/Calibration/interface/EfficiencyPayloads.h"
#include "CondFormats/DataRecord/interface/ExEfficiency.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondCore/CondDB/interface/KeyList.h"

//
#include "CondCore/CondDB/interface/Serialization.h"

namespace cond {
  template <> cond::BaseKeyed* createPayload<cond::BaseKeyed>( const std::string& payloadTypeName ){
    if( payloadTypeName == "condex::ConfI" ) return new condex::ConfI;
    if( payloadTypeName == "condex::ConfF" ) return new condex::ConfF;
    throwException(std::string("Type mismatch, target object is type \"")+payloadTypeName+"\"",
		   "createPayload" );
  }

}

REGISTER_PLUGIN(PedestalsRcd,Pedestals);
REGISTER_PLUGIN(anotherPedestalsRcd,Pedestals);
REGISTER_PLUGIN(mySiStripNoisesRcd,mySiStripNoises);
REGISTER_PLUGIN(ExEfficiencyRcd, condex::Efficiency);
REGISTER_PLUGIN(ExDwarfRcd, cond::BaseKeyed);
// REGISTER_PLUGIN(ExDwarfListRcd, cond::KeyList);
REGISTER_KEYLIST_PLUGIN(ExDwarfListRcd, cond::persistency::KeyList, ExDwarfRcd);
