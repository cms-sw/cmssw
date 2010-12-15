import FWCore.ParameterSet.Config as cms

print "The pile up is taken from 7 TeV MinBias files."

# Take pileup events from files
PileUpSimulatorBlock = cms.PSet(
    PileUpSimulator = cms.PSet(
        # The file with the last minimum bias events read in the previous run
        # to be put in the local running directory (if desired)
        inputFile = cms.untracked.string('PileUpInputFile.txt'),
        # Special files of minimum bias events (generated with 
        # cmsRun FastSimulation/PileUpProducer/test/producePileUpEvents7TeV_cfg.py)
        fileNames = cms.untracked.vstring(
            'MinBias7TeV_001.root', 
            'MinBias7TeV_002.root', 
            'MinBias7TeV_003.root', 
            'MinBias7TeV_004.root', 
            'MinBias7TeV_005.root', 
            'MinBias7TeV_006.root', 
            'MinBias7TeV_007.root', 
            'MinBias7TeV_008.root', 
            'MinBias7TeV_009.root', 
            'MinBias7TeV_010.root'),
        averageNumber = cms.double(0.0)
    )
)

