import FWCore.ParameterSet.Config as cms

process = cms.Process("eg")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(==TOTEV==)
)

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Histograms
process.load("DQMServices.Core.DQM_cfg")

# Input files
process.source = cms.Source(
    "PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    #  
    # Pion 15 GeV
    fileNames = cms.untracked.vstring(
        'file:==INPUTFILE==_0.root', 
        'file:==INPUTFILE==_1.root', 
        'file:==INPUTFILE==_2.root', 
        'file:==INPUTFILE==_3.root', 
        'file:==INPUTFILE==_4.root'
    )
)

process.testNU = cms.EDFilter(
    "testNuclearInteractions",
    TestParticleFilter = cms.PSet(
        # Particles with |eta| > etaMax (momentum direction at primary vertex) 
        # are not simulated 
        etaMax = cms.double(5.0),
        # Charged particles with pT < pTMin (GeV/c) are not simulated
        pTMin = cms.double(0.0),
        # Particles with energy smaller than EMin (GeV) are not simulated
        EMin = cms.double(0.0),
        # Protons with energy in excess of this value (GeV) will kept no matter what
        EProton = cms.double(99999.0)
    ),
    SaveNuclearInteractions = cms.bool(True),
    MaxNumberOfNuclearInteractions = cms.uint32(==MAXNU==),
    NUEventFile = cms.untracked.string('==NUFILE=='),
    OutputFile = cms.untracked.string('==ROOTFILE==')
)

# Famos SimHits 
process.load("FastSimulation.Configuration.CommonInputsFake_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = False

# Path to run what is needed
process.p = cms.Path(
    process.offlineBeamSpot+
    process.famosPileUp+
    process.famosSimHits+
    process.testNU
)


# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.detailedInfo = dict(extension = 'txt')

