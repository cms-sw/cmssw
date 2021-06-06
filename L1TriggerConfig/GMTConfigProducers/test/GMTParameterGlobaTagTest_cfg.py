import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('GMTParameterGlobalTagTest')

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source('EmptySource')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
 
process.GlobalTag.globaltag = 'STARTUP_31X::All'

process.gmtDigis = cms.EDProducer("L1MuGlobalMuonTrigger",
    Debug = cms.untracked.int32(9),
    BX_min = cms.int32(-1),
    BX_max = cms.int32(1),
    BX_min_readout = cms.int32(-1),
    BX_max_readout = cms.int32(1),
    DTCandidates = cms.InputTag("none"),
    RPCbCandidates = cms.InputTag("none"),
    CSCCandidates = cms.InputTag("none"),
    RPCfCandidates = cms.InputTag("none"),
    MipIsoData = cms.InputTag("none"),
    WriteLUTsAndRegs = cms.untracked.bool(False)
)

process.p = cms.Path(process.gmtDigis)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    )
)
