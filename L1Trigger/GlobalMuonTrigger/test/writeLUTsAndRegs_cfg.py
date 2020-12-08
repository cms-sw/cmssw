import FWCore.ParameterSet.Config as cms

# LUT generator process
process = cms.Process("LUTgen")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# just run once
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source('EmptySource')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
 
process.GlobalTag.globaltag = 'GR_R_50_V11::All'
#process.GlobalTag.globaltag = 'GR_R_42_V18::All'

######################
# GMT emulator setup #
######################

# load external parameter data (TODO: Get this from DB as well)
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi")
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff")

# load the GMT simulator 
process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cfi")

# Clear event data
process.gmtDigis.DTCandidates = cms.InputTag("none", "")
process.gmtDigis.CSCCandidates = cms.InputTag("none", "")
process.gmtDigis.RPCbCandidates = cms.InputTag("none", "")
process.gmtDigis.RPCfCandidates = cms.InputTag("none", "")

# GMT emulator debugging
process.gmtDigis.Debug = cms.untracked.int32(9)

process.load('L1TriggerConfig.GMTConfigProducers.L1MuGMTParameters_cfi')
process.L1MuGMTParameters.MergeMethodPtBrl=cms.string("byRank")
process.L1MuGMTParameters.MergeMethodPtFwd=cms.string("byRank")
process.L1MuGMTParameters.VersionSortRankEtaQLUT = cms.uint32(275)

# Tell the emulator to generate LUTs
process.gmtDigis.WriteLUTsAndRegs = cms.untracked.bool(True)

# And run!
process.path = cms.Path(process.gmtDigis)

