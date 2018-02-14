# The following comments couldn't be translated into the new config version:

# replace famosSimHits.SimulateMuons = false
# The digis are needed for the L1 simulation
# replace caloRecHits.RecHitsFactory.doDigis=true

# histos limits and binning

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=180GeV/c2
# include "FastSimulation/Configuration/data/HZZllll.cfi"
# Generate ttbar events
#  include "FastSimulation/Configuration/data/ttbar.cfi"
# Generate multijet events with different ptHAT bins
#  include "FastSimulation/Configuration/data/QCDpt80-120.cfi"
#  include "FastSimulation/Configuration/data/QCDpt600-800.cfi"
# Generate Minimum Bias Events
#  include "FastSimulation/Configuration/data/MinBiasEvents.cfi"
# Generate muons with a flat pT particle gun, and with pT=10.
# include "FastSimulation/Configuration/data/FlatPtMuonGun.cfi"
# replace FlatRandomPtGunProducer.PGunParameters.PartID={130}
# Generate di-electrons with pT=35 GeV
process.load("FastSimulation.Configuration.DiElectrons_cfi")

# Famos sequences (no HLT here)
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Set the early collions 10TeV parameters (as in the standard RelVals)
process.famosSimHits.VertexGenerator.SigmaZ=cms.double(3.8)
process.famosSimHits.VertexGenerator.Emittance = cms.double(7.03e-08)
process.famosSimHits.VertexGenerator.BetaStar = cms.double(300.0)


#     
# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronMCAnalyzer",
    electronCollection = cms.InputTag("pixelMatchGsfElectrons"),
    Nbinxyz = cms.int32(50),
    Nbineop2D = cms.int32(30),
    Nbinp = cms.int32(50),
    Nbineta2D = cms.int32(50),
    Etamin = cms.double(-2.5),
    Nbinfhits = cms.int32(20),
    Dphimin = cms.double(-0.01),
    Pmax = cms.double(300.0),
    Phimax = cms.double(3.2),
    Phimin = cms.double(-3.2),
    Eopmax = cms.double(5.0),
    mcTruthCollection = cms.InputTag("source"),
    # efficiency cuts
    MaxPt = cms.double(100.0),
    Nbinlhits = cms.int32(5),
    Nbinpteff = cms.int32(19),
    Nbinphi2D = cms.int32(32),
    Nbindetamatch2D = cms.int32(50),
    Nbineta = cms.int32(50),
    DeltaR = cms.double(0.05),
    outputFile = cms.string('gsfElectronHistos.root'),
    Nbinp2D = cms.int32(50),
    Nbindeta = cms.int32(100),
    Nbinpt2D = cms.int32(50),
    Nbindetamatch = cms.int32(100),
    Fhitsmax = cms.double(20.0),
    Lhitsmax = cms.double(10.0),
    Nbinphi = cms.int32(64),
    Eopmaxsht = cms.double(3.0),
    MaxAbsEta = cms.double(2.5),
    Nbindphimatch = cms.int32(100),
    Detamax = cms.double(0.005),
    Nbinpt = cms.int32(50),
    Nbindphimatch2D = cms.int32(50),
    Etamax = cms.double(2.5),
    Dphimax = cms.double(0.01),
    Dphimatchmax = cms.double(0.2),
    Detamatchmax = cms.double(0.05),
    Nbindphi = cms.int32(100),
    Detamatchmin = cms.double(-0.05),
    Ptmax = cms.double(100.0),
    Nbineop = cms.int32(50),
    Dphimatchmin = cms.double(-0.2),
    Detamin = cms.double(-0.005)
)

process.Timing = cms.Service("Timing")

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_mix_*_*'),
    fileName = cms.untracked.string('MyFirstFamosFile.root')
)

process.p1 = cms.Path(process.famosWithElectrons*process.gsfElectronAnalysis)
process.outpath = cms.EndPath(process.o1)
process.famosPileUp.PileUpSimulator.averageNumber = 0.0
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.MessageLogger.destinations = ['detailedInfo.txt']


