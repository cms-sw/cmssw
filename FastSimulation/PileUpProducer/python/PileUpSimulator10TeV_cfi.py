import FWCore.ParameterSet.Config as cms

# Take pileup events from files
PileUpSimulatorBlock = cms.PSet(
    PileUpSimulator = cms.PSet(
        # The file with the last minimum bias events read in the previous run
        # to be put in the local running directory (if desired)
        inputFile = cms.untracked.string('PileUpInputFile.txt'),
        # Special files of minimum bias events (generated with 
        # cmsRun FastSimulation/PileUpProducer/test/producePileUpEvents10TeV_cfg.py)
        fileNames = cms.untracked.vstring(
            'MinBias10TeV_001.root', 
            'MinBias10TeV_002.root', 
            'MinBias10TeV_003.root', 
            'MinBias10TeV_004.root', 
            'MinBias10TeV_005.root', 
            'MinBias10TeV_006.root', 
            'MinBias10TeV_007.root', 
            'MinBias10TeV_008.root', 
            'MinBias10TeV_009.root', 
            'MinBias10TeV_010.root'),
        usePoisson = cms.bool(True),
        averageNumber = cms.double(0.0),
        probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24),
        probValue = cms.vdouble(1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    )
)

