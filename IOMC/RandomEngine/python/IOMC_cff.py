import FWCore.ParameterSet.Config as cms

FullSimEngine = cms.untracked.string('MixMaxRng')
FastSimEngine = cms.untracked.string('MixMaxRng')
RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    externalLHEProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(234567),
        engineName = FullSimEngine
    ),
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = FullSimEngine
    ),

    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(98765432),
        engineName = FullSimEngine
    ),
    LHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    hiSignalLHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(88776655),
        engineName = FastSimEngine
    ),
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = FullSimEngine
    ),
    mix = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = FullSimEngine
    ),
    mixData = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = FullSimEngine
    ),
    simSiStripDigiSimLink = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine
    ),
    simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine
    ),
    simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(11223344),
        engineName = FullSimEngine
    ),
    simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine
    ),
#
# HI generation & simulation is a special processing/chain,
# integrated since 330 cycle
#
   hiSignal = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = FullSimEngine
    ),
   hiSignalG4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = FullSimEngine
    ),

#
# FastSim numbers
# integrated since 6.0
#
    famosPileUp = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = FastSimEngine
    ),

    mixGenPU = cms.PSet(
        initialSeed = cms.untracked.uint32(918273), # intentionally the same as famosPileUp
        engineName = FastSimEngine
    ),
    
    mixSimCaloHits = cms.PSet(
         initialSeed = cms.untracked.uint32(918273), 
         engineName = FastSimEngine
    ),     

    mixRecoTracks = cms.PSet(
         initialSeed = cms.untracked.uint32(918273), 
         engineName = FastSimEngine
    ),
                                           
    fastSimProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(13579),
        engineName = FastSimEngine
    ),

    fastTrackerRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(24680),
        engineName = FastSimEngine
    ),

    ecalRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(654321),
        engineName = FastSimEngine
    ),

    ecalPreshowerRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(6541321),
        engineName = FastSimEngine
    ),

    hbhereco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = FastSimEngine
    ),

    horeco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = FastSimEngine
    ),

    hfreco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = FastSimEngine
    ),
    
    paramMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(54525),
        engineName = FastSimEngine
    ),

    l1ParamMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(6453209),
        engineName = FastSimEngine
    ),

    MuonSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(987346),
        engineName = FastSimEngine
    ),
   #CTPPS FastSim
    CTPPSFastRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(1357987),
        engineName = FastSimEngine
     ),
    # filter for simulated beam spot
    simBeamSpotFilter = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = FullSimEngine
    ),

    RPixDetDigitizer = cms.PSet(
        initialSeed = cms.untracked.uint32(137137),
        engineName = FullSimEngine
    ),

    RPSiDetDigitizer = cms.PSet(
        initialSeed = cms.untracked.uint32(137137),
        engineName = FullSimEngine
    )
    # to save the status of the last event (useful for crashes)
    ,saveFileName = cms.untracked.string('')
    # to restore the status of the last event, 
    # comment the line above and decomment the following one
    #   ,restoreFileName = cms.untracked.string('RandomEngineState.log')  
    # to reproduce events using the RandomEngineStateProducer (source excluded),
    # decomment the following one
    #   ,restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
)

randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toModify(RandomNumberGeneratorService, simMuonGEMDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine) )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(
    RandomNumberGeneratorService, 
    simMuonGEMDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine)
)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(
    RandomNumberGeneratorService,
    simMuonME0Digis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine),
    simMuonME0PseudoDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine),
    simMuonME0PseudoReDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(7654321),
        engineName = FullSimEngine),
    simMuonME0PseudoReDigisCoarse = cms.PSet(
        initialSeed = cms.untracked.uint32(2234567),
        engineName = FullSimEngine),
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    RandomNumberGeneratorService,
    trackTimeValueMapProducer = cms.PSet( 
        initialSeed = cms.untracked.uint32(1234567), 
        engineName = FullSimEngine 
        ),
    gsfTrackTimeValueMapProducer = cms.PSet( 
        initialSeed = cms.untracked.uint32(1234567), 
        engineName = FullSimEngine 
        ),
    ecalBarrelClusterFastTimer = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = FullSimEngine
        )
)
