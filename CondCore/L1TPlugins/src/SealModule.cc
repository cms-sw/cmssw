#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondTools/L1Trigger/interface/WriterProxy.h"

#include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

REGISTER_PLUGIN(L1TriggerKeyRcd, L1TriggerKey);
REGISTER_L1_WRITER(L1TriggerKeyRcd, L1TriggerKey);

REGISTER_PLUGIN(L1CSCTPParametersRcd, L1CSCTPParameters);
REGISTER_L1_WRITER(L1CSCTPParametersRcd, L1CSCTPParameters);

REGISTER_PLUGIN(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);
REGISTER_L1_WRITER(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);

REGISTER_PLUGIN(L1JetEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1JetEtScaleRcd, L1CaloEtScale);

REGISTER_PLUGIN(L1EmEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1EmEtScaleRcd, L1CaloEtScale);
