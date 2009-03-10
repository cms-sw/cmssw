import FWCore.ParameterSet.Config as cms

process = cms.Process("TKAN")
#process = cms.Process("TEST")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000)
)


# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# For histograms
process.load("DQMServices.Core.DQM_cfg")

# Input
process.source = cms.Source(
    "PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
        #'file:SinglePion_FastFull_0.root',
        #'file:SinglePion_FastFull_1.root',
        #'file:SinglePion_FastFull_2.root',
        #'file:SinglePion_FastFull_3.root',
        #'file:SinglePion_FastFull_4.root',
        #'file:SinglePion_FastFull_5.root',
        #'file:SinglePion_FastFull_6.root',
        #'file:SinglePion_FastFull_7.root'
        #'file:SinglePion_FastFull_All.root'
        'file:fevt_SinglePion_E0_1.root',
        'file:fevt_SinglePion_E1_1.root',
        'file:fevt_SinglePion_E2_1.root',
        'file:fevt_SinglePion_E3_1.root',
        'file:fevt_SinglePion_E3_2.root',
        'file:fevt_SinglePion_E4_1.root',
        'file:fevt_SinglePion_E4_2.root',
        'file:fevt_SinglePion_E5_1.root',
        'file:fevt_SinglePion_E5_2.root',
        'file:fevt_SinglePion_E5_3.root',
        'file:fevt_SinglePion_E5_4.root',
        'file:fevt_SinglePion_E6_1.root',
        'file:fevt_SinglePion_E6_2.root',
        'file:fevt_SinglePion_E6_3.root',
        'file:fevt_SinglePion_E6_4.root',
        'file:fevt_SinglePion_E7_1.root',
        'file:fevt_SinglePion_E7_2.root',
        'file:fevt_SinglePion_E7_3.root',
        'file:fevt_SinglePion_E7_4.root'
    ),
    noEventSort=cms.untracked.bool(True)
)

process.testTK = cms.EDFilter(
    "testTrackingIterations",
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
    firstFull = cms.InputTag("firstfilter","","PROD"),
    firstFast = cms.InputTag("firstfilter","","TKAN"),
    secondFull = cms.InputTag("secStep","","PROD"),
    secondFast = cms.InputTag("secStep","","TKAN"),
    thirdFull = cms.InputTag("thStep","","PROD"),
    thirdFast = cms.InputTag("thStep","","TKAN")
)

# Famos SimHits 
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.GlobalTag.globaltag = "IDEAL_V9::All"

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Magnetic field
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# No SimHits
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True
process.famosSimHits.TrackerSimHits.pTmin = 0.2

# Path to run what is needed
process.p = cms.Path(
    # Produce fast sim with full sim !
    process.famosWithTracks +
    # Analyse Fast and Full simultaneously
    process.testTK
)

# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['test.txt']

# Should be commented out in the analysis step
process.o1 = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('SinglePion_FastFull.root'),
    outputCommands = cms.untracked.vstring(
       "keep *",
       "drop *_mix_*_*"
    )
)
process.outpath = cms.EndPath(process.o1)
