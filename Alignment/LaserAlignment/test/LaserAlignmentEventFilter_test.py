# EXAMPLE PYTHON CONFIGURATION FILE FOR COMMISSIONING CONVERTER
# for global runs (multi-partition, zero suppressed)
# editable options are marked with a #@@@

import FWCore.ParameterSet.Config as cms

process = cms.Process( "LaserAlignmentEventFilterTest" )

## input files
process.source = cms.Source( "PoolSource",
  fileNames = cms.untracked.vstring(
       '/store/data/Run2010A/TestEnables/RAW/v1/000/140/124/56E00D1B-308F-DF11-BE54-001D09F24691.root'              
    ),


    # LAS Runs, TgBoard Delays 32 - 42
    #eventsToProcess = cms.untracked.VEventRange(
    #'119407:672983000-119407:733700000'
    #)

    # LAS Run, TgBoard Delay 38 + 39
    #eventsToProcess = cms.untracked.VEventRange(
    #'119407:708316000-119407:715494000'
    #)

    #LAS Run, TgBoard Delay 39
    #eventsToProcess = cms.untracked.VEventRange(
    #'119407:713637500-119407:715494000'
    #)

    #LAS Run, TgBoard Delay 39
    #eventsToProcess = cms.untracked.VEventRange(
    #'119407:713500000-119407:715500000'
    #)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 )
)

## message logger
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('LaserAlignmentEventFilter')
)


process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
#process.GlobalTag.globaltag = 'GR09_31X_V5P::All'
process.GlobalTag.globaltag = cms.string('GR_R_37X_V6::All')


process.load('Alignment.LaserAlignment.LaserAlignmentEventFilter_cfi')

process.p = cms.Path( process.LaserAlignmentEventFilter )

