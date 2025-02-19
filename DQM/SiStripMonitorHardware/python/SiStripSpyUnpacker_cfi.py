# Configuration include for unpacking scope mode digis from spy channel data
#============================================================================
import FWCore.ParameterSet.Config as cms

SiStripSpyUnpacker = cms.EDProducer(
    "SiStripSpyUnpackerModule",
    FEDIDs = cms.vuint32(),                     # FED IDs to look at - leave empty for all FEDs
    #FEDIDs = cms.vuint32(50, 187, 260, 356),   # or use a subset.
    InputProductLabel = cms.InputTag('source'),
    AllowIncompleteEvents = cms.bool(True),
    StoreCounters = cms.bool(True),
    StoreScopeRawDigis = cms.bool(True)         # Note - needs to be True for use in other modules.
    )
