import FWCore.ParameterSet.Config as cms

siStripQualityESProducer = cms.ESProducer(
    "SiStripQualityESProducer",
    appendToDataLabel = cms.string(''),
    ListOfRecordToMerge = cms.VPSet(
    cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
    cms.PSet( record = cms.string("RunInfoRcd"),           tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadStripRcd"),   tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") )
    ),
    ReduceGranularity = cms.bool(False),
    # Minumum percentage of bad strips to set the full apv as bad.
    ThresholdForReducedGranularity = cms.double(0.3),
    # True means all the debug output from adding the RunInfo (default is False)
    PrintDebugOutput = cms.bool(False),
    # "True" means that the RunInfo is used even if all the feds are off (including other subdetectors).
    # This means that if the RunInfo was filled with a fake empty object we will still set the full tracker as bad.
    # With "False", instead, in that case the RunInfo information is discarded.
    # Default is "False".
    UseEmptyRunInfo = cms.bool(False),
)


