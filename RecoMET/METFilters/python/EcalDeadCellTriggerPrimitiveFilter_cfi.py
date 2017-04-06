import FWCore.ParameterSet.Config as cms

EcalDeadCellTriggerPrimitiveFilter = cms.EDFilter(
    'EcalDeadCellTriggerPrimitiveFilter',

    # when activated, the filter does not filter event.
    # the filter is however storing a bool in the event, that can be used to take the
    # filtering decision a posteriori
    taggingMode = cms.bool( False ),
    
    debug = cms.bool( False ),
    verbose = cms.int32( 1 ),
    
    tpDigiCollection = cms.InputTag("ecalTPSkimNA"),
    etValToBeFlagged = cms.double(127.49),
    
    ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    
    maskedEcalChannelStatusThreshold = cms.int32( 1 ),
    
    doEEfilter = cms.untracked.bool( True ), # turn it on by default

    useTTsum = cms.bool ( True ),
    usekTPSaturated = cms.bool ( False)
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( EcalDeadCellTriggerPrimitiveFilter, 
    doEEfilter = cms.untracked.bool(False)
)
