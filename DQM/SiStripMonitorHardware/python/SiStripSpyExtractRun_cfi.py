# Configuration include for unpacking scope mode digis from spy channel data
#============================================================================
import FWCore.ParameterSet.Config as cms

SiStripSpyExtractRun = cms.EDProducer(
    "SiStripSpyExtractRunModule",
    RunNumberTag = cms.InputTag('SiStripSpyUnpacker','GlobalRunNumber'),
    OutputTextFile = cms.string('runNumber.txt')
    )
