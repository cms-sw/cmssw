# 091104 SV this is just a prototype for debug issues,
# values are meaningless since luts are different from each TRACO
import FWCore.ParameterSet.Config as cms

LutParametersBlock = cms.PSet(
    LutParameters = cms.PSet(
        Debug = cms.untracked.bool(False), ## Debug flag
    	BTIC = cms.untracked.int32(0), ## BTIC traco parameter
        D = cms.untracked.double(0), 
        XCN = cms.untracked.double(0),
        WHEEL = cms.untracked.int32(-1) ## pos/neg chamber type
    )
)


