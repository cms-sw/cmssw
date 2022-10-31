from __future__ import print_function
from fnmatch import fnmatch
import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing
import sys

from Configuration.StandardSequences.Eras import eras
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_DoubleMuon_string

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

options = VarParsing.VarParsing ()
options.register('maxEvents',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events to process (\"-1\" for all)")
options.register ('era',
                  '2017', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "CMS running era")

options.register ('GlobalTag',
                  '113X_mc2017_realistic_v4', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "seed number")

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

options.register ('myseed',
                  '1', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "seed number")

options.register ('myfile',
                  filesDefaultMC_DoubleMuon_string, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "file name")

options.register ('FileList',
                  '', # default value
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,
                  "FileList in DAS format")

options.register ('outputName',
                  'default', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "output file")

options.parseArguments()

if(options.FileList):
    print("FileList:           ", options.FileList)
else:
    print("inputFile:          ", options.myfile)
print("outputFile:         ", "DiMuonVertexValidation_{fname}_{fseed}.root".format(fname = options.outputName,fseed=options.myseed))
print("era:                ", options.era)
print("conditionGT:        ", options.GlobalTag)
print("conditionOverwrite: ", options.records)
print("external conditions:", options.external)
print("max events:         ", options.maxEvents)

if options.era=='2016':
    print("===> running era 2016")
    process = cms.Process('Analysis',eras.Run2_2016)
elif options.era=='2017':
    print("===> running era 2017")
    process = cms.Process('Analysis',eras.Run2_2017)
elif options.era=='2018':
    print("===> running era 2018")
    process = cms.Process('Analysis',eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

###################################################################
# Tell the program where to find the conditons
connection_map = [
    ('Tracker*', 'frontier://PromptProd/CMS_CONDITIONS'),
    ('SiPixel*', 'frontier://PromptProd/CMS_CONDITIONS'),
    ('SiStrip*', 'frontier://PromptProd/CMS_CONDITIONS'),
    ('Beam*', 'frontier://PromptProd/CMS_CONDITIONS'),
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
        print("control point:",rcd,tag)
        if len(rcd)==0:
            print("no overriding will occur")
            continue
        records.append(
            cms.PSet(
                record = cms.string(rcd),
                tag    = cms.string(tag),
                connect = cms.string(best_match(rcd))
                )
            )

###################################################################
# configure the Global Tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')
process.GlobalTag.toGet = cms.VPSet(*records)

'''
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("TrackerAlignmentRcd"),
             tag = cms.string("TrackerAlignment_Upgrade2017_design_v4"),
             #tag = cms.string("TrackerAlignment_2017_ultralegacymc_v1"),
             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
         ),
    cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"),
             tag = cms.string("TrackerAlignmentErrorsExtended_Upgrade2017_design_v0"),
             #tag = cms.string("TrackerAlignmentExtendedErrors_2017_ultralegacymc_v1"),
             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
         )
)
'''

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

#process.load('FWCore.MessageService.MessageLogger_cfi')
#process.MessageLogger.cerr.FwkReport.reportEvery = 1

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.TrackRefitter=dict()
process.MessageLogger.PrimaryVertexProducer=dict()
process.MessageLogger.DiMuonVertexValidation=dict()
process.MessageLogger.DiLeptonHelpCounts=dict()
process.MessageLogger.PlotsVsKinematics=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(100)
                                   ),                                                      
    DiMuonVertexValidation = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    DiLeptonHelpCounts = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

###################################################################
# Source
###################################################################
if(options.FileList):
    print('Loading file list from ASCII file')
    filelist = FileUtils.loadListFromFile (options.FileList)
    readFiles = cms.untracked.vstring( *filelist)
else:
    readFiles = cms.untracked.vstring([options.myfile])

process.source = cms.Source("PoolSource",
                            fileNames = readFiles,
                            #skipEvents = cms.untracked.uint32(45000)
)

###################################################################
# TransientTrack from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTransientTracks
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')

####################################################################
# Get the BeamSpot
####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

####################################################################
# Track Refitter
####################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.TrackRefitter.src = "generalTracks"
#process.TrackRefitter.src = "ALCARECOTkAlDiMuonVertexTracks"
process.TrackRefitter.TrajectoryInEvent = True
process.TrackRefitter.NavigationSchool = ''
process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

####################################################################
# Sequence
####################################################################
process.seqTrackselRefit = cms.Sequence(process.offlineBeamSpot*
                                        # in case NavigatioSchool is set !=''
                                        #process.MeasurementTrackerEvent*
                                        process.TrackRefitter)

####################################################################
# Re-do vertices
####################################################################
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
process.offlinePrimaryVerticesFromRefittedTrks = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel = cms.InputTag("TrackRefitter")

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",fileName=cms.string("DiMuonVertexValidation_"+options.outputName+"_"+options.myseed+".root"))

# Additional output definition
process.analysis = cms.EDAnalyzer("DiMuonVertexValidation",
                                  useReco = cms.bool(True),
                                  ## the two parameters below are mutually exclusive,
                                  ## depending if RECO or ALCARECO is used
                                  muons  = cms.InputTag('muons'),
                                  #muonTracks = cms.InputTag('ALCARECOTkAlDiMuon'),
                                  tracks = cms.InputTag('TrackRefitter'),
                                  vertices = cms.InputTag('offlinePrimaryVerticesFromRefittedTrks'))

####################################################################
# Path
####################################################################
process.p = cms.Path(process.seqTrackselRefit                        +
                     process.offlinePrimaryVerticesFromRefittedTrks  +
                     process.analysis
                     )
