import FWCore.ParameterSet.Config as cms

source = cms.Source( "DaqSource",
    evtsPerLS = cms.untracked.uint32( 0 ),
    useEventCounter = cms.untracked.bool( False ),
    keepUsingPsidFromTrigger = cms.untracked.bool( False ),
    writeStatusFile = cms.untracked.bool( False ),
    processingMode = cms.untracked.string( "RunsLumisAndEvents" ),
    readerPluginName = cms.untracked.string( "FUShmReader" )
)
