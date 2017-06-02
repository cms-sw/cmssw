import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## global tag
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_GRun', '')

## input file (adapt input file name correspondingly)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:susy_dqm_miniAOD.root"),
)

## number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

## output options
process.options = cms.untracked.PSet(
        Rethrow = cms.untracked.vstring('ProductNotFound'),
        fileMode = cms.untracked.string('FULLMERGE')
)

process.MessageLogger = cms.Service("MessageLogger",
       destinations   = cms.untracked.vstring('detailedInfo','critical','cerr'),
       critical       = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
       detailedInfo   = cms.untracked.PSet(threshold  = cms.untracked.string('INFO')),
       cerr           = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING'))
)

## DQMStore and output configuration
process.DQMStore.collateHistograms        = True
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun  = True
process.dqmSaver.saveByRun      = cms.untracked.int32( -1)
process.dqmSaver.saveAtJobEnd   = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(  1)

##harvesting module
process.susyDQMPostProcessor = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring('Physics/Susy/*'),
        verbose = cms.untracked.uint32(2),
        resolution = cms.vstring(""),
        efficiency = cms.vstring(
        "fractionOfGoodJetsVsEta 'Fraction of jets passing loose quality cuts; #eta; #epsilon' fractionOfGoodJetsVsEta_numerator fractionOfGoodJetsVsEta_denominator",
        "fractionOfGoodJetsVsPhi 'Fraction of jets passing loose quality cuts; #phi; #epsilon' fractionOfGoodJetsVsPhi_numerator fractionOfGoodJetsVsPhi_denominator",
        "electronEfficiencyVsPt 'Loose electron ID efficiency vs p_{T}; gen electron p_{T}; #epsilon' electronEfficiencyVsPt_numerator electronEfficiencyVsPt_denominator",
        "muonEfficiencyVsPt 'Loose muon ID efficiency vs p_{T}; gen muon p_{T}; #epsilon' muonEfficiencyVsPt_numerator muonEfficiencyVsPt_denominator",
        ),
)

## path definitions
process.edmtome = cms.Path(
    process.EDMtoME
)
process.dqmsave = cms.Path(
    process.DQMSaver
)
process.harvesting = cms.Path(
    process.susyDQMPostProcessor
)

## schedule definition
process.schedule = cms.Schedule(process.edmtome,process.harvesting,process.dqmsave)
