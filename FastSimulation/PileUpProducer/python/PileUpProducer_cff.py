import FWCore.ParameterSet.Config as cms

famosPileUp = cms.EDProducer(
    "PileUpProducer",
    # The conditions for pile-up event generation
    PileUpSimulatorBlock = cms.PSet(
        PileUpSimulator = cms.PSet(
            # The file with the last minimum bias events read in the previous run
            # to be put in the local running directory (if desired)
            inputFile = cms.untracked.string('PileUpInputFile.txt'),
            # Special files of minimum bias events (generated with 
            # cmsRun FastSimulation/PileUpProducer/test/producePileUpEvents_cfg.py)
            fileNames = cms.untracked.vstring(
                'MinBias14TeV_001.root', 
                'MinBias14TeV_002.root', 
                'MinBias14TeV_003.root', 
                'MinBias14TeV_004.root', 
                'MinBias14TeV_005.root', 
                'MinBias14TeV_006.root', 
                'MinBias14TeV_007.root', 
                'MinBias14TeV_008.root', 
                'MinBias14TeV_009.root', 
                'MinBias14TeV_010.root'),
            averageNumber = cms.double(0.0)
            )
        ),
    VertexGenerator = cms.PSet(
        type = cms.string('BetaFunc'),
        Phi = cms.double(0.0),
        Y0 = cms.double(0.3929),
        BetaStar = cms.double(70.0),
        Emittance = cms.double(5.86e-08),
        SigmaZ = cms.double(6.16),
        TimeOffset = cms.double(0.0),
        Alpha = cms.double(0.0),
        X0 = cms.double(0.244),
        Z0 = cms.double(0.4145)
        )
    )
