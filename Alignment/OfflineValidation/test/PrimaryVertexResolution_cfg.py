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
    print rcd
    for pattern, string in connection_map:
        print pattern, fnmatch(rcd, pattern)
        if fnmatch(rcd, pattern):
            return string

options = VarParsing.VarParsing("analysis")

options.register ('outputRootFile',
                  "pvresolution_test.root",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "output root file")

options.register ('records',
                  [],
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "record:tag names to be used/changed from GT")

options.register ('external',
                  [],
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "record:fle.db picks the following record from this external file")

options.register ('GlobalTag',
                  'auto:run2_data',
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.parseArguments()

print "conditionGT       : ", options.GlobalTag
print "conditionOverwrite: ", options.records
print "external conditions:", options.external
print "outputFile        : ", options.outputRootFile

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1000) # every 100th only
        #    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load('Configuration.Geometry.GeometryRecoDB_cff')

process.load('Configuration/StandardSequences/Services_cff')
process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/express/Run2017E/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/303/832/00000/02800A3F-31A1-E711-8DD1-02163E011E71.root',
        '/store/express/Run2017E/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/303/832/00000/04F52DB5-E3A1-E711-A96E-02163E01390D.root',
        '/store/express/Run2017E/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/303/832/00000/062DACC5-E3A1-E711-A664-02163E01A36F.root',
        '/store/express/Run2017E/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/303/832/00000/06A8A611-41A1-E711-B399-02163E019B55.root',
        '/store/express/Run2017E/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/303/832/00000/083C9E2E-E1A1-E711-80A0-02163E01A31D.root',
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
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minPixelLayersWithHits        = 2   

process.PrimaryVertexResolution = cms.EDAnalyzer('PrimaryVertexResolution',
                                                 vtxCollection       = cms.InputTag("offlinePrimaryVerticesFromRefittedTrks"),
                                                 trackCollection     = cms.InputTag("TrackRefitter"),		
                                                 minVertexNdf        = cms.untracked.double(10.),
                                                 minVertexMeanWeight = cms.untracked.double(0.5)
                                                 )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outputRootFile),	
                                   closeFileFast = cms.untracked.bool(False)
                                   )

process.p = cms.Path(process.offlineBeamSpot                        + 
                     process.TrackRefitter                          + 
                     process.offlinePrimaryVerticesFromRefittedTrks +
                     process.PrimaryVertexResolution)


