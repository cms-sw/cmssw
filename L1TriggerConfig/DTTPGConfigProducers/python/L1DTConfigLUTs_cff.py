# 091104 SV this is just a prototype for debug issues
import FWCore.ParameterSet.Config as cms

LutParametersBlock = cms.PSet(
    LutParameters = cms.PSet(
        Debug = cms.untracked.bool(False), ## Debug flag
    	BTIC = cms.untracked.int32(31), ## BTIC traco parameter
        D = cms.untracked.double(66.5), 
        XCN = cms.untracked.double(80.2),
        WHEEL = cms.untracked.int32(-1) ## pos/neg chamber type
    )
)


