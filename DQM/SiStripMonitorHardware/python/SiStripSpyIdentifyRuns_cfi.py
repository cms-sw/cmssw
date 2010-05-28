# Configuration include for unpacking scope mode digis from spy channel data
#============================================================================
import FWCore.ParameterSet.Config as cms

SiStripSpyIdentifyRuns = cms.EDProducer(
    "SiStripSpyIdentifyRunsModule",
    InputProductLabel = cms.InputTag('source'),
    OutputTextFile = cms.string('spyRunNumbers.txt')
    )
