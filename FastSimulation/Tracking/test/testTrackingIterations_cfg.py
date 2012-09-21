import FWCore.ParameterSet.Config as cms

process = cms.Process("TKAN")
#process = cms.Process("TEST")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet(
###    input = cms.untracked.int32(5000)
    input = cms.untracked.int32(-1)
)

###process.Tracer = cms.Service("Tracer")


# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# For histograms
process.load("DQMServices.Core.DQM_cfg")

# Input
process.source = cms.Source(
    "PoolSource",
##    debugFlag = cms.untracked.bool(True),
##    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
##    'file:fevt.root'
##    'file:test.root'
##        #'file:SinglePion_FastFull_0.root',
##        #'file:SinglePion_FastFull_1.root',
##        #'file:SinglePion_FastFull_2.root',
##        #'file:SinglePion_FastFull_3.root',
##        #'file:SinglePion_FastFull_4.root',
##        #'file:SinglePion_FastFull_5.root',
##        #'file:SinglePion_FastFull_6.root',
##        #'file:SinglePion_FastFull_7.root'
##        'file:SinglePion_FastFull_All.root'
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E0_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E1_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E2_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E3_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E3_2.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E4_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E4_2.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E5_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E5_2.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E5_3.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E5_4.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E6_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E6_2.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E6_3.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E6_4.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E7_1.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E7_2.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E7_3.root',
    'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SinglePion_E7_4.root'
##      'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E0_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E1_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E2_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E3_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E3_2.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E4_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E4_2.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E5_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E5_2.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E5_3.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E5_4.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E6_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E6_2.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E6_3.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E6_4.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E7_1.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E7_2.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E7_3.root',
##     'rfio:/castor/cern.ch/user/a/azzi/CMSSW350pre2/fevt_SingleK0s_E7_4.root'
   ),
    noEventSort=cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
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
    zeroFull = cms.InputTag("zeroStepHighPurity","","PROD"),
    zeroFast = cms.InputTag("zeroStepFilter","","TKAN"),
    firstFull = cms.InputTag("firstStepHighPurity","","PROD"),
    firstFast = cms.InputTag("firstfilter","","TKAN"),
    secondFull = cms.InputTag("secfilter","","PROD"),
    secondFast = cms.InputTag("secStep","","TKAN"),
    thirdFull = cms.InputTag("thfilter","","PROD"),
    thirdFast = cms.InputTag("thStep","","TKAN"),
##    fourthFull = cms.InputTag("fourthfilter","","PROD"),
    fourthFull = cms.InputTag("fourthStepHighPurity","","PROD"),
    fourthFast = cms.InputTag("fouStep","","TKAN"),
    fifthFull = cms.InputTag("fifthStepHighPurity","","PROD"),
    fifthFast = cms.InputTag("fifthStep","","TKAN"),
)

# Famos SimHits 
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.GlobalTag.globaltag = "MC_3XY_V15::All"
##process.GlobalTag.globaltag = "STARTUP_31X::All"

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
    process.famosWithTracks *
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
#process.outpath = cms.EndPath(process.o1)

##process.load('logger_cfi')
###process.MessageLogger._moduleCanTalk('iterativeFirstSeeds')
####process.MessageLogger._moduleCanTalk('iterativeSecondSeeds')
#####process.MessageLogger._moduleCanTalk('iterativeFifthSeeds')
#####process.MessageLogger._moduleCanTalk('iterativeFifthTrackCandidatesWithPairs')
#####process.MessageLogger._moduleCanTalk('iterativeFifthTrackMerging')
#####process.MessageLogger._moduleCanTalk('iterativeFifthTrackFiltering')


####process.MessageLogger._moduleCanTalk('iterativeFirstTracks')
####process.MessageLogger._moduleCanTalk('iterativeFirstTracksWithPairs')
###process.MessageLogger._moduleCanTalk('iterativeSecondTracks')
###process.MessageLogger._moduleCanTalk('iterativeSecondTracksWithTriplets')
##process.MessageLogger._moduleCanTalk('iterativeThirdTracks')
##process.MessageLogger._moduleCanTalk('iterativeThirdTracksWithPairs')
#####process.MessageLogger._moduleCanTalk('iterativeFourthTracks')
#####process.MessageLogger._moduleCanTalk('iterativeFourthTracksWithPairs')
#####process.MessageLogger._moduleCanTalk('iterativeFifthTracks')
#####process.MessageLogger._moduleCanTalk('iterativeFifthTracksWithPairs')

##process.MessageLogger._categoryCanTalk('TrackProducer')
##process.MessageLogger._categoryCanTalk('TrackFitters')
##process.MessageLogger._categoryCanTalk('FastTracking')

