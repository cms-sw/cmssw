import FWCore.ParameterSet.Config as cms

# Take pileup events from files
PileUpSimulatorBlock = cms.PSet(
    PileUpSimulator = cms.PSet(
        # The file with the last minimum bias events read in the previous run
        # to be put in the local running directory (if desired)
        inputFile = cms.untracked.string('PileUpInputFile.txt'),
        # Special files of minimum bias events (generated with 
        # cmsRun FastSimulation/PileUpProducer/test/producePileUpEvents14TeV_cfg.py)
        fileNames = cms.untracked.vstring(
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_001.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_002.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_003.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_004.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_005.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_006.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_007.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_008.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_009.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias_010.root'),
        usePoisson = cms.bool(True),
        averageNumber = cms.double(0.0),
        probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24),
        probValue = cms.vdouble(1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    )
)

