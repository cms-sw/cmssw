# The following comments couldn't be translated into the new config version:

# Set random seeds. 

import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    l1ParamMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(6453209),
        engineName = cms.untracked.string('TRandom3')
    ),
    simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(524964),
        engineName = cms.untracked.string('TRandom3')
    ),
    caloRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    # This is to initialize the random engines used for  Famos
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('TRandom3')
    ),
    # To save the status of the last event (useful for crashes)
    # Just give a name to the file you want the status to be saved
    # otherwise just put saveFileName = ""
    saveFileName = cms.untracked.string(''),
    famosPileUp = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    # To restore the status of the last event, just un-comment the
    # following line (and comment the saveFileName line!)
    # untracked string restoreFileName = "RandomEngineState.log"
    # To reproduce events using the RandomEngineStateProducer (source
    # excluded), comment the sourceSeed definition, and un-comment 
    # the restoreStateLabel
    # untracked string restoreStateLabel = "randomEngineStateProducer"
    # This is to initialize the random engine of the source
    theSource = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('TRandom3')
    ),
    simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(525432),
        engineName = cms.untracked.string('TRandom3')
    ),
    famosSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(13579),
        engineName = cms.untracked.string('TRandom3')
    ),
    paramMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(54525),
        engineName = cms.untracked.string('TRandom3')
    ),
    MuonSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(987346),
        engineName = cms.untracked.string('TRandom3')
    ),
    simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(67673876),
        engineName = cms.untracked.string('TRandom3')
    ),
    siTrackerGaussianSmearingRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(24680),
        engineName = cms.untracked.string('TRandom3')
    )
)


