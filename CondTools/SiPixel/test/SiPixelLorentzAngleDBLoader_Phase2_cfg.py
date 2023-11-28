#  for phase2

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
import sys

process = cms.Process("SiPixelLorentzAngleLoader",eras.Phase2)

import FWCore.ParameterSet.VarParsing as opts
opt = opts.VarParsing ('analysis')

opt.register('geometry',          'T25',
             opts.VarParsing.multiplicity.singleton, 
             opts.VarParsing.varType.string,
             'Tracker Geometry Default = T15')

opt.register('isEmpty',  False,
             opts.VarParsing.multiplicity.singleton, 
             opts.VarParsing.varType.bool,
             'If True, produce empty payload forWidth')

opt.parseArguments()

tGeometry = opt.geometry
if tGeometry == 'T5':
    geometry_cff = 'GeometryExtended2023D17_cff'
    recoGeometry_cff = 'GeometryExtended2023D17Reco_cff'
    LA_value = 0.106
    tag = 'SiPixelLorentzAngle_Phase2_T5' 
    
elif tGeometry == 'T6':
    geometry_cff = 'GeometryExtended2023D35_cff'
    recoGeometry_cff = 'GeometryExtended2023D35Reco_cff'
    LA_value = 0.106
    tag = 'SiPixelLorentzAngle_Phase2_T6' 
    
elif tGeometry == 'T11':
    geometry_cff = 'GeometryExtended2023D29_cff'
    recoGeometry_cff = 'GeometryExtended2023D29Reco_cff'
    LA_value = 0.106
    tag = 'SiPixelLorentzAngle_Phase2_T11' 
    
elif tGeometry == 'T14':
    geometry_cff = 'GeometryExtended2023D41_cff'
    recoGeometry_cff = 'GeometryExtended2023D41Reco_cff'
    LA_value = 0.106
    tag = 'SiPixelLorentzAngle_Phase2_T14' 
    
elif tGeometry == 'T15':
    geometry_cff = 'GeometryExtended2023D42_cff'
    recoGeometry_cff = 'GeometryExtended2023D42Reco_cff'
    LA_value = 0.0503
    tag = 'SiPixelLorentzAngle_Phase2_T15' 

elif tGeometry == 'T25':
    geometry_cff = 'GeometryExtended2026D97_cff'
    recoGeometry_cff = 'GeometryExtended2026D97Reco_cff'
    LA_value = 0.0503
    tag = 'SiPixelLorentzAngle_Phase2_T25_v1'
     
else:
    print("Unknown tracker geometry")
    print("What are you doing ?!?!?!?!")
    exit(1)
    
if opt.isEmpty:
    LA_value = 0
    tag += '_forWidthEmpty'
    
sqlite_file = 'sqlite_file:' + tag + '.db'
geometry_cff = 'Configuration.Geometry.' + geometry_cff
recoGeometry_cff = 'Configuration.Geometry.' + recoGeometry_cff


process.load(recoGeometry_cff)
process.load(geometry_cff)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

##### DATABASE CONNNECTION AND INPUT TAGS ######
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
    authenticationPath = cms.untracked.string('.'),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(1),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(0),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    timetype = cms.untracked.string('runnumber'),    
    connect = cms.string(sqlite_file),        
    toPut = cms.VPSet(
        cms.PSet(
            #record = cms.string('SiPixelLorentzAngleSimRcd'),
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string(tag)         
        ),
    )
)

###### LORENTZ ANGLE OBJECT ######
process.SiPixelLorentzAngle = cms.EDAnalyzer(
    "SiPixelLorentzAngleDBLoader",
    
    # enter -9999 if individual input
    bPixLorentzAnglePerTesla = cms.untracked.double(-9999),
    fPixLorentzAnglePerTesla = cms.untracked.double(-9999),    

    #in case of PSet (only works if above is -9999)
    # One common value for BPix for now
    BPixParameters = cms.untracked.VPSet(
        cms.PSet(layer = cms.int32(1), angle = cms.double(0.0)),
        cms.PSet(layer = cms.int32(2), angle = cms.double(0.053)),
        cms.PSet(layer = cms.int32(3), angle = cms.double(0.053)),
        cms.PSet(layer = cms.int32(4), angle = cms.double(0.053)),
    ),
    FPixParameters = cms.untracked.VPSet(
        cms.PSet(angle = cms.double(0.0) 
             ),
    ),    
    # List of Exceptions
    ModuleParameters = cms.untracked.VPSet(
    ),

    useFile = cms.bool(False),
    #record = cms.untracked.string('SiPixelLorentzAngleRcd'),  
    #record = cms.untracked.string('SiPixelLorentzAngleSimRcd'),  
    fileName = cms.string('lorentzFit.txt')	
)

process.p = cms.Path(
    #    process.SiPixelLorentzAngleSim
    process.SiPixelLorentzAngle
)

