import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

##
## Message Logger
##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100) # every 100th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))
process.MessageLogger.statistics.append('cout')

##
## Conditions
##
#process.load("Configuration.StandardSequences.FakeConditions_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = 'IDEAL_V5::All' # 'CRUZET4_V3P::All'

##
## Geometry
##
process.load("Configuration.StandardSequences.Geometry_cff")

##
## Magnetic Field
##
process.load("Configuration.StandardSequences.MagneticField_cff")
# 0 T:
#process.localUniform = cms.ESProducer("UniformMagneticFieldESProducer",
#                                      ZFieldInTesla = cms.double(0.0)
#                                      )
#process.prefer_localUniform = cms.ESPrefer("UniformMagneticFieldESProducer",
#                                           "localUniform")
# 0 T from CMSSW_2_1_5 on:
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")

##
## BeamSpot from database (i.e. GlobalTag), needed for Refitter
##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

##
## Load DBSetup (if needed)
##
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
process.trackerAlignment = poolDBESSource.clone()
process.trackerAlignment.connect = 'frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'
process.trackerAlignment.toGet = cms.VPSet(cms.PSet(
    record = cms.string('TrackerAlignmentRcd'),
    tag = cms.string('TrackerIdealGeometry210_mc') 
), 
    cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        # tag = cms.string('TrackerIdealGeometryErrors210_mc') # for APE = 0
        tag = cms.string('TrackerSurveyLASOnlyScenarioErrors210_mc')
    ))
process.prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")


##
## Input File(s)
##
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    ##
    ## 21X RelVal Samples, please replace accordingly
    ##
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/06D7B343-365C-DD11-AD13-001617E30D00.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/08056306-445C-DD11-A35C-000423D992A4.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/1C1712DA-365C-DD11-870A-001617C3B79A.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/1CA078CB-375C-DD11-96D2-001617E30D4A.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/2282BAF4-345C-DD11-83BF-001617C3B710.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/2CA50990-3B5C-DD11-8305-000423D6C8E6.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/3014096D-375C-DD11-A2C0-001617E30F4C.root'
    ##
    ## CRUZET 4 ALCARECO (on CAF)
    ##
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/000690AE-E771-DD11-93D6-000423D996C8.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/002C355B-3372-DD11-A93C-000423D98DD4.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/00FD0190-7A71-DD11-A95F-000423D98844.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/0237113C-1F71-DD11-95B9-000423D8FA38.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/024501B6-CE71-DD11-88F6-000423D98930.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/043D4A5B-1E71-DD11-8C00-001D09F2441B.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/048E64E3-9371-DD11-8C2C-001D09F2AF1E.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/04E932C2-A471-DD11-8CEC-001D09F2924F.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/06075384-6171-DD11-975E-001617C3B65A.root' #,...
    )
)

##
## Maximum number of Events
##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

##
## Output File Configuration
##
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('AlignmentValidation.root')
)

##
## Load and Configure track selection for alignment
##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
# process.AlignmentTrackSelector.src = "ALCARECOTkAlCosmicsCTF0T" e.g. for cosmics ALCARECO
process.AlignmentTrackSelector.applyBasicCuts = False #True
#process.AlignmentTrackSelector.ptMin   = .3

##
## Load and Configure TrackRefitter
##
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True

##
## Load and Configure OfflineValidation
##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_cfi")
process.TrackerOfflineValidation.Tracks = 'TrackRefitter'
process.TrackerOfflineValidation.trajectoryInput = 'TrackRefitter'
process.TrackerOfflineValidation.moduleLevelHistsTransient = True
process.TrackerOfflineValidation.TH1ResModules = cms.PSet(
    xmin = cms.double(-0.5),
    Nbinx = cms.int32(300),
    xmax = cms.double(0.5)
)
process.TrackerOfflineValidation.TH1NormResModules = cms.PSet(
    xmin = cms.double(-3.0),
    Nbinx = cms.int32(300),
    xmax = cms.double(3.0)
)


##
## PATH
##
process.p = cms.Path(process.offlineBeamSpot
                     *process.AlignmentTrackSelector
                     *process.TrackRefitter
                     *process.TrackerOfflineValidation
                     )
