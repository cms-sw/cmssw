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
    etValToBeFlagged = cms.double(127.5),
    
    ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    
    maskedEcalChannelStatusThreshold = cms.int32( 1 ),
    
    doEEfilter = cms.untracked.bool( True ), # turn it on by default
    
    makeProfileRoot = cms.untracked.bool( False ),
    profileRootName = cms.untracked.string("deadCellFilterProfile.root" ),

)
