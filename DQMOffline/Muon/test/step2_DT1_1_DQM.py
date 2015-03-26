# Auto generated configuration file
# using: 
# Revision: 1.341 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2_DT1_1 -s DQM -n 1000 --eventcontent DQM --conditions auto:com10 --geometry Ideal --filein /store/data/Run2011B/SingleMu/RECO/PromptReco-v1/000/179/410/F0B03882-45FE-E011-9609-001D09F25393.root --data --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('DQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/data/Run2011B/SingleMu/RECO/PromptReco-v1/000/179/411/069A7031-41FE-E011-9BCA-003048D3733E.root') #,
#                                       '/store/data/Run2011B/SingleMu/RECO/PromptReco-v1/000/179/411/0C68A99B-47FE-E011-9275-003048F1C836.root',
#                                       '/store/data/Run2011B/SingleMu/RECO/PromptReco-v1/000/179/411/2C16F59A-47FE-E011-AC81-003048F024FA.root',
#                                       '/store/data/Run2011B/SingleMu/RECO/PromptReco-v1/000/179/411/3EE243D2-47FE-E011-992C-003048F24A04.root',
#                                       '/store/data/Run2011B/SingleMu/RECO/PromptReco-v1/000/179/411/565F8ED5-46FE-E011-A831-003048F117EC.root')
)

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('step2_DT1_1 nevts:1000'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition
process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('step2_DT1_1_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition
# process.MessageLogger = cms.Service("MessageLogger",
#     debugModules = cms.untracked.vstring('*'),
#     threshold = cms.untracked.string('INFO'),
#     destinations = cms.untracked.vstring('MuonDQM_DEBUG.log')
# )
                   
# Other statements
process.GlobalTag.globaltag = 'GR_R_50_V0::All'

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(
       record  = cms.string( 'AlCaRecoTriggerBitsRcd' ),
       tag     = cms.string( 'MuonDQMTrigger' ),
       label   = cms.untracked.string( 'MuonDQMTrigger'),
       connect = cms.untracked.string( 'sqlite_file:MuonDQMTrigger.db')
    )
)

process.AlCaRecoTriggerBitsRcdRead = cms.EDAnalyzer( "AlCaRecoTriggerBitsRcdRead"
, outputType  = cms.untracked.string( 'text' )
, rawFileName = cms.untracked.string( 'MuonDQMTrigger_test' )
)

process.p = cms.Path(
  process.AlCaRecoTriggerBitsRcdRead
)

# Path and EndPath definitions
process.dqmoffline_step = cms.Path(process.DQMOfflinePOG)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(#process.p,
                                process.dqmoffline_step,
                                process.endjob_step,
                                process.DQMoutput_step)

