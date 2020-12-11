import FWCore.ParameterSet.Config as cms

process = cms.Process("GMTPATT")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Message Logger
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )
)
    
process.source = cms.Source("L1MuGMTHWFileReader",
    fileNames = cms.untracked.vstring("file:gmt_testfile.h4mu.1000.dat")
)
process.gmtPattern = cms.EDAnalyzer("L1MuGMTPattern",
    GMTInputTag = cms.untracked.InputTag("gmtDigis"),
    OutputFile = cms.untracked.string("gmt_testfile.dat"),
    OutputType = cms.untracked.int32(1)
)

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
 
process.GlobalTag.globaltag = 'START42_V13::All'
#process.GlobalTag.globaltag = 'GR_R_42_V13::All'

######################
# GMT emulator setup #
######################

# load external parameter data (TODO: Get this from DB as well)
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi")
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff")

# load the GMT simulator 
process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cfi")

process.gmtDigis.DTCandidates = "source:DT"
process.gmtDigis.CSCCandidates = 'source:CSC'
process.gmtDigis.RPCbCandidates = 'source:RPCb'
process.gmtDigis.RPCfCandidates = 'source:RPCf'
process.gmtDigis.MipIsoData = 'source'
process.gmtDigis.Debug = 0
process.gmtDigis.BX_min = -1
process.gmtDigis.BX_max = 1
process.gmtDigis.BX_min_readout = -1
process.gmtDigis.BX_max_readout = 1
process.gmtDigis.SendMipIso = cms.untracked.bool(True)

#process.L1MuGMTParameters.SubsystemMask = 0

process.load('L1TriggerConfig.GMTConfigProducers.L1MuGMTParameters_cfi')
process.L1MuGMTParameters.MergeMethodPtBrl=cms.string("byRank")
process.L1MuGMTParameters.MergeMethodPtFwd=cms.string("byRank")
process.L1MuGMTParameters.VersionSortRankEtaQLUT = cms.uint32(275)

process.p = cms.Path(process.gmtDigis * process.gmtPattern)
