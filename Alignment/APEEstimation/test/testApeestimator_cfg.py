from __future__ import print_function
import os

import FWCore.ParameterSet.Config as cms




##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('sample', 'wlnu', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Input sample")
options.register('isTest', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Test run")

# get and parse the command line arguments
if( hasattr(sys, "argv") ):
    for args in sys.argv :
        arg = args.split(',')
        for val in arg:
            val = val.split('=')
            if(len(val)==2):
                setattr(options,val[0], val[1])

print("Input sample: ", options.sample)
print("Test run: ", options.isTest)



##
## Process definition
##
process = cms.Process("ApeEstimator")



##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.SectorBuilder=dict()
process.MessageLogger.ResidualErrorBinning=dict()
process.MessageLogger.HitSelector=dict()
process.MessageLogger.CalculateAPE=dict()
process.MessageLogger.ApeEstimator=dict()
#process.MessageLogger.TrackRefitter=dict()
process.MessageLogger.AlignmentTrackSelector=dict()
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = -1  # Do not use =0, else all error messages (except those listed below) are supressed
process.MessageLogger.cerr.SectorBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.HitSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.CalculateAPE = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.ApeEstimator = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.AlignmentTrackSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 ## really show only every 1000th



##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)



##
## Input sample definition
##
isData1 = isData2 = False
isData = False
isQcd = isWlnu = isZmumu = isZtautau = isZmumu10 = isZmumu20 = False
isMc = False
if options.sample == 'data1':
    isData1 = True
    isData = True
elif options.sample == 'data2':
    isData2 = True
    isData = True
elif options.sample == 'qcd':
    isQcd = True
    isMc = True
elif options.sample == 'wlnu':
    isWlnu = True
    isMc = True
elif options.sample == 'zmumu':
    isZmumu = True
    isMc = True
elif options.sample == 'ztautau':
    isZtautau = True
    isMc = True
elif options.sample == 'zmumu10':
    isZmumu10 = True
    isMc = True
elif options.sample == 'zmumu20':
    isZmumu20 = True
    isMc = True
else:
    print('ERROR --- incorrect data sammple: ', options.sample)
    exit(8888)



##
## Input Files
##
if isData1:
    process.load("Alignment.APEEstimation.samples.Data_TkAlMuonIsolated_Run2011A_May10ReReco_ApeSkim_cff")
elif isData2:
    process.load("Alignment.APEEstimationsamples.Data_TkAlMuonIsolated_Run2011A_PromptV4_ApeSkim_cff")
elif isQcd:
    process.load("Alignment.APEEstimation.samples.Mc_TkAlMuonIsolated_Summer11_qcd_ApeSkim_cff")
elif isWlnu:
    process.load("Alignment.APEEstimation.samples.Mc_WJetsToLNu_74XTest_ApeSkim_cff")
elif isZmumu10:
    process.load("Alignment.APEEstimation.samples.Mc_TkAlMuonIsolated_Summer11_zmumu10_ApeSkim_cff")
elif isZmumu20:
    process.load("Alignment.APEEstimation.samples.Mc_TkAlMuonIsolated_Summer11_zmumu20_ApeSkim_cff")
    


##
## Number of Events (should be after input file)
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
if options.isTest: process.maxEvents.input = 10001


##
## Check run and event numbers for Dublicates --- only for real data
##
#process.source.duplicateCheckMode = cms.untracked.string("noDuplicateCheck")
#process.source.duplicateCheckMode = cms.untracked.string("checkEachFile")
process.source.duplicateCheckMode = cms.untracked.string("checkEachRealDataFile")
#process.source.duplicateCheckMode = cms.untracked.string("checkAllFilesOpened")   # default value



##
## Whole Refitter Sequence
##
process.load("Alignment.APEEstimation.TrackRefitter_38T_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')


##### To be used when running on Phys14MC with a CMSSW version > 72X
#process.GlobalTag.toGet = cms.VPSet(
#    cms.PSet(
#        record = cms.string("BeamSpotObjectsRcd"),
#        tag = cms.string("Realistic8TeVCollisions_START50_V13_v1_mc"),
#        connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS"),
#    )
#) 
print("Using global tag "+process.GlobalTag.globaltag._value)



##
## New pixel templates
##
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(
        record = cms.string("SiPixelTemplateDBObjectRcd"),
        tag = cms.string("SiPixelTemplateDBObject_38T_v3_mc"),
        connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS"),
    )
) 



##
## Alignment and APE
##
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
## Choose Alignment (w/o touching APE)
if isMc:
  process.myTrackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'frontier://FrontierProd/CMS_CONDITIONS', # or your sqlite file
    toGet = [
      cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry210_mc') # 'TrackerAlignment_2009_v2_offline'
      ),
    ],
  )
  process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")

process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")
if isData:
  # Recent geometry
  process.myTrackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'frontier://FrontierProd/CMS_CONDITIONS',
    toGet = [
      cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerAlignment_GR10_v6_offline'),
      ),
    ],
  )
  process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")
  # Kinks and bows
  process.myTrackerAlignmentKinksAndBows = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'frontier://FrontierProd/CMS_CONDITIONS',
    toGet = [
      cms.PSet(
        record = cms.string('TrackerSurfaceDeformationRcd'),
        tag = cms.string('TrackerSurfaceDeformations_v1_offline'),
      ),
    ],
  )
  process.es_prefer_trackerAlignmentKinksAndBows = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentKinksAndBows")

## APE (set to zero)
process.myTrackerAlignmentErr = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'frontier://FrontierProd/CMS_CONDITIONS',
    toGet = [
      cms.PSet(
        record = cms.string('TrackerAlignmentErrorExtendedRcd'),
        tag = cms.string('TrackerIdealGeometryErrorsExtended210_mc')
      ),
    ],
)
process.es_prefer_trackerAlignmentErr = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentErr")



##
## Trigger Selection
##
process.load("Alignment.APEEstimation.TriggerSelection_cff")



##
## ApeEstimator
##
from Alignment.APEEstimation.ApeEstimator_cff import *
process.ApeEstimator1 = ApeEstimator.clone(
    #~ tjTkAssociationMapTag = "TrackRefitterHighPurityForApeEstimator",
    tjTkAssociationMapTag = "TrackRefitterForApeEstimator",
    maxTracksPerEvent = 0,
    applyTrackCuts = False,
    Sectors = RecentSectors,
    analyzerMode = False,
    calculateApe = True
)
process.ApeEstimator1.HitSelector.width = []
process.ApeEstimator1.HitSelector.maxIndex = []
process.ApeEstimator1.HitSelector.widthProj = []
process.ApeEstimator1.HitSelector.widthDiff = []
process.ApeEstimator1.HitSelector.edgeStrips = []
process.ApeEstimator1.HitSelector.sOverN = []
process.ApeEstimator1.HitSelector.maxCharge = []
process.ApeEstimator1.HitSelector.chargeOnEdges = []
process.ApeEstimator1.HitSelector.probX = []
process.ApeEstimator1.HitSelector.phiSensX = []
process.ApeEstimator1.HitSelector.phiSensY = []
process.ApeEstimator1.HitSelector.errXHit = []
process.ApeEstimator1.HitSelector.chargePixel = []
process.ApeEstimator1.HitSelector.widthX = []
process.ApeEstimator1.HitSelector.widthY = []
process.ApeEstimator1.HitSelector.logClusterProbability = []
process.ApeEstimator1.HitSelector.isOnEdge = []
process.ApeEstimator1.HitSelector.qBin = []


process.ApeEstimator2 = process.ApeEstimator1.clone(
    Sectors = ValidationSectors,
    analyzerMode = True,
    calculateApe = False,
)

process.ApeEstimator3 = process.ApeEstimator2.clone(
    zoomHists = False,
)



##
## Output File Configuration
##
outputName = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/'
if options.isTest:
    outputName = outputName + 'test_'
outputName = outputName + options.sample + '.root'

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(outputName),
    closeFileFast = cms.untracked.bool(True)
)



##
## Path
##
process.p = cms.Path(
    process.TriggerSelectionSequence*
    process.RefitterHighPuritySequence*
    (process.ApeEstimator1+
     process.ApeEstimator2+
     process.ApeEstimator3
    )
)



