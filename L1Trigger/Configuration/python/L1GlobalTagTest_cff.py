#
# cff file defining sequences to print the content of 
# L1 trigger records from a global tag 
#
# V M Ghete 2009-03-04


import FWCore.ParameterSet.Config as cms

# Regional Calorimeter Trigger
#

l1RCTParametersTest = cms.EDAnalyzer('L1RCTParametersTester')
l1RCTChannelMaskTest = cms.EDAnalyzer('L1RCTChannelMaskTester')
l1RCTOutputScalesTest = cms.EDAnalyzer('L1ScalesTester')

printGlobalTagL1Rct = cms.Sequence(l1RCTParametersTest*l1RCTChannelMaskTest*l1RCTOutputScalesTest)

# Global Calorimeter Trigger
#

from L1TriggerConfig.GctConfigProducers.l1GctConfigDump_cfi import l1GctConfigDump
from L1TriggerConfig.L1GeometryProducers.l1CaloGeometryDump_cfi import l1CaloGeometryDump
printGlobalTagL1Gct = cms.Sequence(l1GctConfigDump*l1CaloGeometryDump)
#printGlobalTagL1Gct = L1TriggerConfig.GctConfigProducers.l1GctConfigDump_cfi.l1GctConfigDump.clone()

# DT TPG
#

# MISSING
#printGlobalTagL1DtTPG = cms.Sequence()

# DT TF
#

DTExtLutTester = cms.EDAnalyzer('DTExtLutTester')
DTPhiLutTester = cms.EDAnalyzer('DTPhiLutTester')
DTPtaLutTester = cms.EDAnalyzer('DTPtaLutTester')
DTEtaPatternLutTester = cms.EDAnalyzer('DTEtaPatternLutTester')
DTQualPatternLutTester = cms.EDAnalyzer('DTQualPatternLutTester')
DTTFParametersTester = cms.EDAnalyzer('DTTFParametersTester')
DTTFMasksTester = cms.EDAnalyzer('DTTFMasksTester')

printGlobalTagL1DtTF = cms.Sequence(DTExtLutTester
                                    *DTPhiLutTester
                                    *DTPtaLutTester
                                    *DTEtaPatternLutTester
                                    *DTQualPatternLutTester
                                    *DTTFParametersTester
                                    *DTTFMasksTester
                                    )

# CSC TF
#
CSCTFParametersTester = cms.EDAnalyzer("L1MuCSCTFParametersTester")

printGlobalTagL1CscTF = cms.Sequence(CSCTFParametersTester)

# RPC Trigger
#
dumpL1RPCConfig = cms.EDAnalyzer('DumpL1RPCConfig',
          fileName = cms.string('PrintGlobalTag_L1RPCConfig.log'))
dumpConeDefinition = cms.EDAnalyzer('DumpConeDefinition',
          fileName = cms.string('PrintGlobalTag_L1RPCConeDefinition.log'))
dumpL1RPCBxOrConfig = cms.EDAnalyzer("DumpL1RPCBxOrConfig")
dumpL1RPCHsbConfig = cms.EDAnalyzer("DumpL1RPCHsbConfig")

printGlobalTagL1Rpc = cms.Sequence(dumpL1RPCConfig*dumpConeDefinition*dumpL1RPCBxOrConfig*dumpL1RPCHsbConfig)

# Global Muon Trigger
#

printL1GmtParameters = cms.EDProducer('L1MuGlobalMuonTrigger',
    Debug = cms.untracked.int32(9),
    BX_min = cms.int32(-1),
    BX_max = cms.int32(1),
    BX_min_readout = cms.int32(-1),
    BX_max_readout = cms.int32(1),
    DTCandidates = cms.InputTag('none'),
    RPCbCandidates = cms.InputTag('none'),
    CSCCandidates = cms.InputTag('none'),
    RPCfCandidates = cms.InputTag('none'),
    MipIsoData = cms.InputTag('none'),
    WriteLUTsAndRegs = cms.untracked.bool(False)
)

printL1GmtMuScales = cms.EDAnalyzer('L1MuScalesTester')

printGlobalTagL1Gmt = cms.Sequence(printL1GmtParameters*printL1GmtMuScales)


# Global Trigger
#
from L1TriggerConfig.L1GtConfigProducers.L1GtTester_cff import *

# all L1 records
printGlobalTagL1 = cms.Sequence(printGlobalTagL1Rct
                                *printGlobalTagL1Gct
#                                *printGlobalTagL1DtTPG
                                *printGlobalTagL1DtTF
                                *printGlobalTagL1CscTF
                                *printGlobalTagL1Rpc
                                *printGlobalTagL1Gmt
                                *printGlobalTagL1Gt
                                )


