import FWCore.ParameterSet.Config as cms

process = cms.Process("TKAN")
#process = cms.Process("TEST")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

###process.Tracer = cms.Service("Tracer")


# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# For histograms
process.load("DQMServices.Core.DQM_cfg")

## # Input
process.source = cms.Source(
    "PoolSource",
  #  debugFlag = cms.untracked.bool(True),
  #  debugVebosity = cms.untracked.uint32(10),
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
## ##        'file:SinglePion_FastFull_All.root'
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
# '/store/relval/CMSSW_3_5_0_pre5/RelValSingleMuPt100/GEN-SIM-RECO/MC_3XY_V20-v1/0008/0E27D097-550E-DF11-8A0D-0030487A322E.root',
#       '/store/relval/CMSSW_3_5_0_pre5/RelValSingleMuPt100/GEN-SIM-RECO/MC_3XY_V20-v1/0007/30D8606B-E70D-DF11-A1DB-001617E30E28.root'
    ),
    noEventSort=cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
    )


#process.generalTracksHighPurity = cms.EDFilter("QualityFilter",
#                                               TrackQuality = cms.string('highPurity'),
#                                               recTracks = cms.InputTag("generalTracks")
#                                               )


process.testTK = cms.EDFilter(
    "testGeneralTracks",
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
    Full = cms.InputTag("generalTracksHighPurity","","PROD"),
    ##    Full = cms.InputTag("generalTracksHighPurity","","TKAN"),
    ##   Full = cms.InputTag("generalTracks","","HLT"),
    Fast = cms.InputTag("generalTracks","","TKAN"),
    )

# Famos SimHits 
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.GlobalTag.globaltag = "MC_3XY_V15::All"

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Magnetic field
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# No SimHits
process.fastSimProducer.SimulateCalorimetry = False

# Path to run what is needed
process.p = cms.Path(
    # Produce fast sim with full sim !
    process.famosWithTracks *
    # Analyse Fast and Full simultaneously
    process.testTK
)

# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.test = dict(extension = 'txt')

# Should be commented out in the analysis step
process.o1 = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('SinglePion_FastFull.root'),
    outputCommands = cms.untracked.vstring(
       "keep *",
       "drop *_mix_*_*"
    )
)
##process.outpath = cms.EndPath(process.o1)

##process.load('logger_cfi')

##process.MessageLogger._moduleCanTalk('iterativeFirstTracksCandidateWithPairs')
##process.MessageLogger._moduleCanTalk('iterativeFirstTracksCandidateWithTriplets')

##process.MessageLogger._moduleCanTalk('iterativeFirstTracksWithPairs')
##process.MessageLogger._moduleCanTalk('iterativeFirstTracksWithTriplets')

##process.MessageLogger._moduleCanTalk('iterativeSecondTrackCandidatesWithTriplets')
##process.MessageLogger._moduleCanTalk('iterativeSecondTracksWithTriplets')

##process.MessageLogger._moduleCanTalk('iterativeThirdTrackCandidatesWithPairs')
##process.MessageLogger._moduleCanTalk('iterativeThirdTracksWithPairs')

##process.MessageLogger._moduleCanTalk('iterativeFourthTrackCandidatesWithPairs')
##process.MessageLogger._moduleCanTalk('iterativeFourthTracksWithPairs')

##process.MessageLogger._moduleCanTalk('iterativeFifthTrackCandidatesWithPairs')
##process.MessageLogger._moduleCanTalk('iterativeFifthTracksWithPairs')

##process.MessageLogger._categoryCanTalk('TrackProducer')
##process.MessageLogger._categoryCanTalk('TrackFitters')
##process.MessageLogger._categoryCanTalk('FastTracking')

