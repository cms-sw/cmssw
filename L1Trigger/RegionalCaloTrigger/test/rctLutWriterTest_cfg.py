import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITELUTS")
# 
# # used to get ECAL scales for RC LUTs
# # ecal tpg params
# es_module = EcalTrigPrimESProducer {
# #untracked string DatabaseFile = "TPG.txt"
# untracked string DatabaseFile = "TPG_RCT_identity.txt"
# }
# 
# # Sources of records
# es_source tpparams6 = EmptyESSource {
# string recordName = "EcalTPGLutGroupRcd"
# vuint32 firstValid = { 1 }
# bool iovIsRunNotTime = true
# }
# es_source tpparams7 = EmptyESSource {
# string recordName = "EcalTPGLutIdMapRcd"
# vuint32 firstValid = { 1 }
# bool iovIsRunNotTime = true
# }
# es_source tpparams12 = EmptyESSource {
# string recordName = "EcalTPGPhysicsConstRcd"
# vuint32 firstValid = { 1 }
# bool iovIsRunNotTime = true
# } 
#include "FWCore/MessageService/data/MessageLogger.cfi"
# configuration of RCT
#include "L1TriggerConfig/RCTConfigProducers/data/L1RCTConfig.cff"
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")

#include "L1Trigger/RegionalCaloTrigger/data/L1RCTTestAnalyzer.cfi"
#replace L1RCTTestAnalyzer.showRegionSums = false
# need to get HCAL scales for RC and JSC (HF) LUTs
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

#replace CaloTPGTranscoder.hcalLUT2 = "TPGcalcDecompress2Identity.txt"
# need for EIC LUT -- em cand linear energy -> rank conversion
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.l1RctParamsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RCTParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.RCTConfigProducers = cms.ESProducer("RCTConfigProducers",
    eGammaHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0),
    eMaxForFGCut = cms.double(-999.0),
    noiseVetoHB = cms.bool(False),
    eMaxForHoECut = cms.double(-999.0),
    hOeCut = cms.double(999.0),
    eGammaECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0),
    eMinForHoECut = cms.double(999.0),
    jscQuietThresholdBarrel = cms.uint32(3),
    hActivityCut = cms.double(999.0),
    eActivityCut = cms.double(999.0),
    jetMETHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0),
    noiseVetoHEplus = cms.bool(False),
    eicIsolationThreshold = cms.uint32(0),
    jetMETLSB = cms.double(1.0),
    jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0),
    eMinForFGCut = cms.double(999.0),
    eGammaLSB = cms.double(1.0),
    jscQuietThresholdEndcap = cms.uint32(3),
    hMinForHoECut = cms.double(999.0),
    noiseVetoHEminus = cms.bool(False)
)

process.l1RctMaskRcds = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RCTChannelMaskRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.rctLutWriter = cms.EDAnalyzer("L1RCTLutWriter",
    useDebugTpgScales = cms.bool(True),
    key = cms.string('noKey')
)

process.p = cms.Path(process.rctLutWriter)
process.schedule = cms.Schedule(process.p)

process.CaloTPGTranscoder.hcalLUT2 = 'L1Trigger/RegionalCaloTrigger/test/data/TPGcalcDecompress2Identity.txt'
process.l1CaloScales.L1CaloEmEtScaleLSB = 1.0


