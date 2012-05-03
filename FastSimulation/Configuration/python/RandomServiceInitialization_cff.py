# The following comments couldn't be translated into the new config version:

# Set random seeds. 

import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service(

    "RandomNumberGeneratorService",

    # To save the status of the last event (useful for crashes)
    # Just give a name to the file you want the status to be saved
    # otherwise just put saveFileName = ""
    saveFileName = cms.untracked.string(''),

    # To restore the status of the last event, just un-comment the
    # following line (and comment the saveFileName line!)
    # restoreFileName = cms.string('RandomEngineState.log'),

    # To reproduce events using the RandomEngineStateProducer (source
    # excluded), comment the sourceSeed definition, and un-comment 
    # the restoreStateLabel
    # restoreStateLabel = cms.string('randomEngineStateProducer'),

    # This is to initialize the random engine of the source
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),

    # This is to initialize the random engines used for  Famos
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('TRandom3')
    ),


    famosPileUp = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),

    famosSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(13579),
        engineName = cms.untracked.string('TRandom3')
    ),

    siTrackerGaussianSmearingRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(24680),
        engineName = cms.untracked.string('TRandom3')
    ),

    ecalRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(654321),
        engineName = cms.untracked.string('TRandom3')
    ),

    ecalPreshowerRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(6541321),
        engineName = cms.untracked.string('TRandom3')
    ),

    hbhereco = cms.PSet(
    initialSeed = cms.untracked.uint32(541321),
    engineName = cms.untracked.string('TRandom3')
    ),

    horeco = cms.PSet(
    initialSeed = cms.untracked.uint32(541321),
    engineName = cms.untracked.string('TRandom3')
    ),

    hfreco = cms.PSet(
    initialSeed = cms.untracked.uint32(541321),
    engineName = cms.untracked.string('TRandom3')
    ),
    
    paramMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(54525),
        engineName = cms.untracked.string('TRandom3')
    ),

    l1ParamMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(6453209),
        engineName = cms.untracked.string('TRandom3')
    ),

    MuonSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(987346),
        engineName = cms.untracked.string('TRandom3')
    ),

    simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(524964),
        engineName = cms.untracked.string('TRandom3')
    ),

    simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(525432),
        engineName = cms.untracked.string('TRandom3')
    ),

    simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(67673876),
        engineName = cms.untracked.string('TRandom3')
    )

)

randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")
