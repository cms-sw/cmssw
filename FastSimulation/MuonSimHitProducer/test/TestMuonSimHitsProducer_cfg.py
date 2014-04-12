# The following comments couldn't be translated into the new config version:

# Just for fun: STAMuonAnalyzer is a simple analyzer of stand-alone muons
# include  "RecoMuon/StandAloneMuonProducer/test/STAMuonAnalyzer.cfi"
# replace STAMuonAnalyzer.DataType = "RealData"
# (not yet adapted to FastSim Simhits... )

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Famos sequences (Frontier conditions)
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FastSimulation.Configuration.CommonInputsFake_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)
process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(10.01),
        MinPt = cms.untracked.double(9.99),
        # you can request more than 1 particle
        #untracked vint32  PartID = { 211, 11, -13 }
        PartID = cms.untracked.vint32(13,13,13,13),
        MaxEta = cms.untracked.double(1.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(1.4),
        MinPhi = cms.untracked.double(-3.14159265359) ## it must be in radians

    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    firstRun = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test_mu.root')
)

process.p1 = cms.Path(process.famosWithTracksAndMuons)
process.outpath = cms.EndPath(process.o1)
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True
process.MessageLogger.destinations = ['detailedInfo.txt']


