# The following comments couldn't be translated into the new config version:

# Special files of minimum bias events (generated with 
# cmsRun FastSimulation/PileUpProducer/test/producePileUpEvents.cfg)

import FWCore.ParameterSet.Config as cms

# Take pileup events from files
PileUpSimulator = cms.PSet(
    # The file with the last minimum bias events read in the previous run
    # to be put in the local running directory (if desired)
    inputFile = cms.untracked.string('PileUpInputFile.txt'),
    fileNames = cms.untracked.vstring('MinBias_001.root', 'MinBias_002.root', 'MinBias_003.root', 'MinBias_004.root', 'MinBias_005.root', 'MinBias_006.root', 'MinBias_007.root', 'MinBias_008.root', 'MinBias_009.root', 'MinBias_010.root'),
    averageNumber = cms.double(0.0)
)

