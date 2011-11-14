import FWCore.ParameterSet.Config as cms

process = cms.Process('TOPDQM')

## imports of standard configurations
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## global tag (needed for JEC)
process.GlobalTag.globaltag = 'GR10_P_V7::All' ## for data with CMSSW_3_6_1_patch4
#process.GlobalTag.globaltag = 'START38_V7::All' ## for CMSSW_3_8_0


## input file(s) for testing
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     #'file:/afs/desy.de/user/r/rwolf/cms13/samples/847D00B0-608E-DF11-A37D-003048678FA0.root' ## for testing at DESY only!!
      '/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/1A19B479-BA3A-DF11-8E43-0017A4770410.root'                                  
    )
)

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

## output
process.output = cms.OutputModule("PoolOutputModule",
  fileName       = cms.untracked.string('topDQM_production.root'),
  outputCommands = cms.untracked.vstring('drop *_*_*_*','keep *_*_*_TOPDQM'),
  splitLevel     = cms.untracked.int32(0),
  dataset = cms.untracked.PSet(
    dataTier   = cms.untracked.string(''),
    filterName = cms.untracked.string('')
  )
)

## load jet corrections
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.prefer("ak5CaloL2L3")

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TopSingleLeptonDQM'   )
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('TopDiLeptonOfflineDQM')
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))


## path definitions
process.p      = cms.Path(
   #process.content *
    process.topDiLeptonOfflineDQM      +
    process.topSingleLeptonDQM         +
    process.topSingleMuonLooseDQM      +    
    process.topSingleMuonMediumDQM     +
    process.topSingleElectronLooseDQM  +    
    process.topSingleElectronMediumDQM
)
process.endjob = cms.Path(
    process.endOfProcess
)
process.fanout = cms.EndPath(
    process.output
)

## schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.endjob,
    process.fanout
)
