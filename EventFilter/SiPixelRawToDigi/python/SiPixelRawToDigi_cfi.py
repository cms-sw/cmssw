import FWCore.ParameterSet.Config as cms

siPixelDigis = cms.EDProducer("SiPixelRawToDigi",
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.bool(True),
    InputLabel = cms.InputTag("siPixelRawData"),
    CheckPixelOrder = cms.bool(False),
    UseQualityInfo = cms.bool(False),
    UseCablingTree = cms.untracked.bool(True),
## ErrorList: list of error codes used by tracking to invalidate modules
    ErrorList = cms.vint32(29),
## UserErrorList: list of error codes used by Pixel experts for investigation
    UserErrorList = cms.vint32(40)
)


## regional seeded unpacking for specialized HLT paths 
siPixelDigisRegional = cms.EDProducer( "SiPixelRawToDigi",
    Timing = cms.untracked.bool( False ),
    IncludeErrors = cms.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    CheckPixelOrder = cms.bool( False ),
    UseQualityInfo = cms.bool( False ),
    UseCablingTree = cms.untracked.bool( True ),
    ErrorList = cms.vint32( ),
    UserErrorList = cms.vint32( ),
    Regions = cms.PSet(
        inputs = cms.VInputTag( "hltL2EtCutDoublePFIsoTau45Trk5" ),
        deltaPhi = cms.vdouble( 0.5 ),
        maxZ = cms.vdouble( 24. ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
    )
)

