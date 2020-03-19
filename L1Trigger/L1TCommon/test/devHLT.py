# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --conditions auto:run2_mc_25ns14e33_v4 -s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval25ns --datatier GEN-SIM-DIGI-RAW-HLTDEBUG -n 10 --era Run2_25ns --eventcontent FEVTDEBUGHLT --filein filelist:step1_dasquery.log --fileout file:step2.root
import FWCore.ParameterSet.Config as cms


#from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
#process = cms.Process('HLT',Run2_25ns)
from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
process = cms.Process('HLT',Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger = cms.Service("MessageLogger",
            destinations = cms.untracked.vstring( 'detailedInfo', 'critical'),
            detailedInfo = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG')),
            debugModules = cms.untracked.vstring( 'hltL1T' )
)
#process.MessageLogger.cerr.FwkReport.reportEvery = 10 # only report every 10th event start
#process.MessageLogger.cerr_stats.threshold = 'INFO' # also info in statistics
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
#process.load('HLT_25ns14e33_v4_hacked_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root', 
#        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/1E9D9F9B-467C-E511-85B6-0025905A6090.root'), 
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/AA4FBC07-3E7C-E511-B9FC-00261894386C.root'), 
#        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/E2072991-3E7C-E511-803D-002618943947.root', 
#        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/FAE20D9D-467C-E511-AF39-0025905B85D8.root'),
    inputCommands = cms.untracked.vstring('keep *', 
        'drop *_genParticles_*_*', 
        'drop *_genParticlesForJets_*_*', 
        'drop *_kt4GenJets_*_*', 
        'drop *_kt6GenJets_*_*', 
        'drop *_iterativeCone5GenJets_*_*', 
        'drop *_ak4GenJets_*_*', 
        'drop *_ak7GenJets_*_*', 
        'drop *_ak8GenJets_*_*', 
        'drop *_ak4GenJetsNoNu_*_*', 
        'drop *_ak8GenJetsNoNu_*_*', 
        'drop *_genCandidatesForMET_*_*', 
        'drop *_genParticlesForMETAllVisible_*_*', 
        'drop *_genMetCalo_*_*', 
        'drop *_genMetCaloAndNonPrompt_*_*', 
        'drop *_genMetTrue_*_*', 
        'drop *_genMetIC5GenJs_*_*'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# additional tests:
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-HLTDEBUG'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# HLT minimal testing:

process.hltL1T = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36 AND L1_SingleEG10" ),
    saveTags = cms.bool( True ),
    L1GtObjectMapTag = cms.InputTag( "simGtStage2Digis" ),
#    L1UseL1TriggerObjectMaps = cms.bool( True ),
#    L1UseAliasesForSeeding = cms.bool( True ),
#    L1GtReadoutRecordTag = cms.InputTag( "simGtStage2Digis" ),
#    L1NrBxInEvent = cms.int32( 3 ),
#    L1TechTriggerSeeding = cms.bool( False )
)

process.HLTTesting  = cms.Sequence( process.hltL1T )


# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_25ns14e33_v4', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.debug_step = cms.Path(process.HLTTesting)
process.dump_step = cms.Path(process.dumpED)
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)







# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.debug_step)
#process.schedule.extend(process.HLTSchedule)
#process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
process.schedule.extend([process.endjob_step])

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
#from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforFullSim 

#call to customisation function customizeHLTforFullSim imported from HLTrigger.Configuration.customizeHLTforMC
#process = customizeHLTforFullSim(process)

# End of customisation functions

