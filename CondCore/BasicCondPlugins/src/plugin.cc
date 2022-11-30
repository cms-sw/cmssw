/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Marco Musich on 30/11/2022.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/BasicPayloadRcd.h"
#include "CondFormats/Common/interface/BasicPayload.h"
#include "CondCore/CondDB/interface/Serialization.h"

REGISTER_PLUGIN(BasicPayloadRcd, cond::BasicPayload);
