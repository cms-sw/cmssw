import FWCore.ParameterSet.Config as cms

siTrackerGaussianSmearingRecHits = cms.EDProducer("SiTrackerGaussianSmearingRecHitConverter",

    # If you want to have RecHits == PSimHits (tracking with PSimHits)
    trackingPSimHits = cms.bool(False),
    trackingPSimHitsEqualSmearing = cms.bool(True),
#vectors: one value per layer (this is ok for phase2
# these are used for subdet>2
#    GeneralResX = cms.double(0.003 ),
#    GeneralResY = cms.double(0.03 ),
# these are used for subdet<=2 (per layer) --> in phase 2 the whole tracker is subdet=1 or 2 only (barrel and forward)
#    GeneralBarResX = cms.vdouble(0.001,0.001,0.001,0.001, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003 ),
#    GeneralBarResY = cms.vdouble(0.002, 0.002, 0.002,0.002, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 ),
#    GeneralFwdResX = cms.vdouble(0.001,0.001,0.001,0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003 ),
#    GeneralFwdResY = cms.vdouble(0.002, 0.002, 0.002, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0),
#vectors: one value per layer (mimicking what was before
    GeneralResX = cms.double(0.003 ),
    GeneralResY = cms.double(3.00 ),
    GeneralBarResX = cms.vdouble(0.001,0.001,0.001,0.001, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003 ),
    GeneralBarResY = cms.vdouble(0.002,0.002,0.002,0.002, 3.000,3.000,3.000,3.000,3.000,3.000),
    GeneralFwdResX = cms.vdouble(0.001,0.001,0.001,0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003 ),
    GeneralFwdResY = cms.vdouble(0.002,0.002,0.002,3.000,3.000,3.000,3.000,3.000,3.000,3.000),

    #converting energy loss from GeV to ADC counts
    GevPerElectron = cms.double(3.61e-09),
    ElectronsPerADC = cms.double(250.0),
             
    doRecHitMatching = cms.bool(True),# if False, I see no tracks anymore... why?

    # Set to (True) for taking the existence of dead modules into account:
    killDeadChannels = cms.bool(False),
    #
    DeltaRaysMomentumCut = cms.double(0.5),

    UseSigma = cms.bool(True),

    VerboseLevel = cms.untracked.int32(2),

    ROUList = cms.VInputTag(cms.InputTag("mix","famosSimHitsTrackerHits")),

    templateIdBarrel = cms.int32( 40 ),
    templateIdForward  = cms.int32( 41 ),


)

