import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('HLTriggerOffline.SUSYBSM.SUSYBSM_postProcessor_cff')


## global tag
process.GlobalTag.globaltag = 'GR_R_52_V7::All'



##process.GlobalTag.globaltag = 'GR_R_38X_V13::All'
#process.GlobalTag.globaltag = 'GR10_P_V10::All'
## input file (adapt input file name correspondingly)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:output/qcd470to600.root"),
    #fileNames = cms.untracked.vstring("file:/tmp/pablom/OutputJason.root"),
    processingMode = cms.untracked.string('RunsAndLumis')
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
## DQMStore and output configuration
process.DQMStore.collateHistograms = True
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = True
process.dqmSaver.saveByRun = cms.untracked.int32( -1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32( 1)


process.MessageLogger = cms.Service("MessageLogger",
        destinations = cms.untracked.vstring('cout'),
        categories = cms.untracked.vstring('HLTMuonVal'),
        debugModules = cms.untracked.vstring('*'),
        threshold = cms.untracked.string('INFO'),
        HLTMuonVal = cms.untracked.PSet(
            #threshold = cms.untracked.string('DEBUG'),
        limit = cms.untracked.int32(100000),
        ),
      )


#process.endpath = cms.EndPath(process.dqmSaver)



## path definitions
process.edmtome = cms.Path(
    process.EDMtoME
    )

#process.SusyExoPostVal = cms.Sequence(process.SUSY_HLT_alphaT_POSTPROCESSING )
#process.SusyExoPostVal = cms.Sequence(process.SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING + process.SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING)

process.susypost = cms.Path(
    process.SusyExoPostVal
    )

process.dqmsave = cms.Path(
    process.DQMSaver
    )
## schedule definition
process.schedule = cms.Schedule(process.edmtome,process.susypost,process.dqmsave)
