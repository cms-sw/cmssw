# The following comments couldn't be translated into the new config version:

# FED id

# Branch name

# FED id

# Branch name

import FWCore.ParameterSet.Config as cms

hcalTBWriter = cms.EDAnalyzer("HcalTBWriter",
    fedRawDataCollectionTag = cms.InputTag('rawDataCollector'),
    # Pattern for output filenames (%d will be replaced by run number)
    FilenamePattern = cms.untracked.string('/data/spool/HTB_%06d.root'),
    # Map of FED-ids to Branch names for the writer
    ChunkNames = cms.untracked.VPSet(cms.PSet(
        Number = cms.untracked.int32(1),
        Name = cms.untracked.string('HCAL_Trigger')
    ), 
        cms.PSet(
            Number = cms.untracked.int32(20),
            Name = cms.untracked.string('HCAL_DCC020')
        ))
)


