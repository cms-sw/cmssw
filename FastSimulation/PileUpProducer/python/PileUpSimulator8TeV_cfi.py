import FWCore.ParameterSet.Config as cms

# Take pileup events from files
PileUpSimulatorBlock = cms.PSet(
    PileUpSimulator = cms.PSet(
        # The file with the last minimum bias events read in the previous run
        # to be put in the local running directory (if desired)
        inputFile = cms.untracked.string('PileUpInputFile.txt'),
        # Special files of minimum bias events (generated with 
        # cmsRun FastSimulation/PileUpProducer/test/producePileUpEvents8TeV_cfg.py)
        fileNames = cms.untracked.vstring(
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_001.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_002.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_003.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_004.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_005.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_006.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_007.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_008.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_009.root', 
            'file:/afs/cern.ch/cms/data/CMSSW/FastSimulation/PileUpProducer/data/MinBias8TeV_010.root'),
        usePoisson = cms.bool(True),
        averageNumber = cms.double(0.0),
#        probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24),
#        probValue = cms.vdouble(0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0698146584,0.0630151648,0.0526654164,0.0402754482,0.0292988928,0.0194384503,0.0122016783,0.007207042,0.004003637,0.0020278322,0.0010739954,0.0004595759,0.0002229748,0.0001028162,4.58337152809607E-05)
    )
)

