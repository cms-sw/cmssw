#! /bin/env cmsRun

'''
cfg to produce ntuples for error scale calibration
here doing refit of tracks and vertices using latest alignment 
'''

import FWCore.ParameterSet.Config as cms
from fnmatch import fnmatch
import FWCore.ParameterSet.VarParsing as VarParsing
from pdb import set_trace

process = cms.Process("PrimaryVertexResolution")

###################################################################
def best_match(rcd):
###################################################################
    '''
    find out where to best match the input conditions
    '''
    print(rcd)
    for pattern, string in connection_map:
        print(pattern, fnmatch(rcd, pattern))
        if fnmatch(rcd, pattern):
            return string

options = VarParsing.VarParsing("analysis")

options.register ('outputRootFile',
                  "pvresolution_test.root",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "output root file")

options.register ('records',
                  [],
                  VarParsing.VarParsing.multiplicity.list,       # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "record:tag names to be used/changed from GT")

options.register ('external',
                  [],
                  VarParsing.VarParsing.multiplicity.list,       # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "record:fle.db picks the following record from this external file")

options.register ('GlobalTag',
                  'auto:run2_data',
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.parseArguments()

print("conditionGT       : ", options.GlobalTag)
print("conditionOverwrite: ", options.records)
print("external conditions:", options.external)
print("outputFile        : ", options.outputRootFile)
print("maxEvents         : ", options.maxEvents)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1000) # every 100th only
        #    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.Geometry.GeometryRecoDB_cff')

process.load('Configuration/StandardSequences/Services_cff')
process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/039/00000/4C851925-AF8D-E811-96A4-02163E010E90.root',
        '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/040/00000/A8A78033-B18D-E811-9477-02163E019F55.root',
        '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/058/00000/6A45BA9D-F88D-E811-B907-FA163E600F07.root',
        # '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/040/00000/B09C5228-B18D-E811-9271-FA163E573834.root',
        # '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/040/00000/C28F8F68-B18D-E811-BC8A-FA163EE8669D.root',
        # '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/040/00000/12BDA2F4-BC8D-E811-BEF2-02163E00C3F8.root',
        # '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/040/00000/788386DF-BC8D-E811-AE14-FA163EC41EB1.root',
        # '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/040/00000/608E8A35-B18D-E811-93ED-FA163E8674D5.root',
        # '/store/express/Run2018C/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/320/040/00000/E03C67DD-BC8D-E811-BBF6-FA163EF274DA.root',
    )
)

###################################################################
# Tell the program where to find the conditons
connection_map = [
    ('Tracker*', 'frontier://PromptProd/CMS_CONDITIONS'),
    ]

if options.external:
    connection_map.extend(
        (i.split(':')[0], 'sqlite_file:%s' % i.split(':')[1]) for i in options.external
        )

connection_map.sort(key=lambda x: -1*len(x[0]))

###################################################################
# creat the map for the GT toGet
records = []
if options.records:
    for record in options.records:
        rcd, tag = tuple(record.split(':'))
        records.append(
            cms.PSet(
                record = cms.string(rcd),
                tag    = cms.string(tag),
                connect = cms.string(best_match(rcd))
                )
            )

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')
process.GlobalTag.toGet = cms.VPSet(*records)
#process.GlobalTag.DumpStat = cms.untracked.bool(True)

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
# remove the following lines if you run on RECO files
process.TrackRefitter.src = 'ALCARECOTkAlMinBias'
process.TrackRefitter.NavigationSchool = ''

## PV refit
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices 
process.offlinePrimaryVerticesFromRefittedTrks  = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel                                       = cms.InputTag("TrackRefitter") 
process.offlinePrimaryVerticesFromRefittedTrks.vertexCollections.maxDistanceToBeam              = 1
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxNormalizedChi2             = 20
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minSiliconLayersWithHits      = 5
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Significance             = 5.0 
# as it was prior to https://github.com/cms-sw/cmssw/commit/c8462ae4313b6be3bbce36e45373aa6e87253c59
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Error                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxDzError                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minPixelLayersWithHits        = 2   

###################################################################
# The trigger filter module
###################################################################
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.HLTFilter = triggerResultsFilter.clone(
    #triggerConditions = cms.vstring("HLT_ZeroBias_*"),
    triggerConditions = cms.vstring("HLT_HT*"),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag( "" ),
    throw = cms.bool(False)
)
###################################################################
# The analysis module
###################################################################
process.myanalysis = cms.EDAnalyzer("GeneralPurposeTrackAnalyzer",
                                    TkTag  = cms.InputTag('TrackRefitter'),
                                    isCosmics = cms.bool(False)
                                    )

###################################################################
# The PV resolution module
###################################################################
process.PrimaryVertexResolution = cms.EDAnalyzer('SplitVertexResolution',
                                                 storeNtuple         = cms.bool(True),
                                                 vtxCollection       = cms.InputTag("offlinePrimaryVerticesFromRefittedTrks"),
                                                 trackCollection     = cms.InputTag("TrackRefitter"),		
                                                 minVertexNdf        = cms.untracked.double(10.),
                                                 minVertexMeanWeight = cms.untracked.double(0.5),
                                                 runControl = cms.untracked.bool(True),
                                                 runControlNumber = cms.untracked.vuint32(320040)
                                                 )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outputRootFile),	
                                   closeFileFast = cms.untracked.bool(False)
                                   )

process.p = cms.Path(process.HLTFilter                               +
                     process.offlineBeamSpot                         +
                     process.TrackRefitter                           +
                     process.offlinePrimaryVerticesFromRefittedTrks  +
                     process.PrimaryVertexResolution                 +
                     process.myanalysis
                     )


