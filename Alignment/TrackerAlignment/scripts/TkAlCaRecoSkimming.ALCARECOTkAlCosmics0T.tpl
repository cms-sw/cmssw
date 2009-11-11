import FWCore.ParameterSet.Config as cms

process = cms.Process("ALCASKIMTkAl")


##__________________________Messages & Convenience____________________________________-
process.load("FWCore.MessageService.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
       destinations = cms.untracked.vstring('LOGFILE_CosmicsSkimmed_<JOB>'),
       statistics   = cms.untracked.vstring('LOGFILE_CosmicsSkimmed_<JOB>'),
       categories   = cms.untracked.vstring('Alignment','AlcaRecoAnalyzer',''),
       debugModules = cms.untracked.vstring( '*' ),

    LOGFILE_Cosmics0T  = cms.untracked.PSet(
           threshold = cms.untracked.string('DEBUG'), 
           INFO      = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
           WARNING   = cms.untracked.PSet( limit = cms.untracked.int32(1000) ),
           ERROR     = cms.untracked.PSet( limit = cms.untracked.int32(1000) ),
           DEBUG     = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
           Alignment = cms.untracked.PSet( limit = cms.untracked.int32(1000) )
       )
   )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

##__________________________________Source_____________________________________________
process.source = cms.Source("PoolSource",
	useCSA08Kludge = cms.untracked.bool(True),
        skipEvents = cms.untracked.uint32(<INIEVT>),                 
   	fileNames = cms.untracked.vstring(
      	  'rfio:/castor/cern.ch/cms/<INPATH>'
     )
    )

##_________________________________Includes____________________________________________
from CondCore.DBCommon.CondDBSetup_cfi import *

process.load('Configuration.EventContent.EventContent_cff')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#--- initialize magnetic field B=0T
###from MagnecticField.Engine.uniformMagneticField_cfg import *
###process.UniformMagneticFieldESProducer.ZFieldInTesla = 0.0 
###es_prefer_UniformMagneticFieldESProducer = cms.ESPrefer("UniformMagneticFieldESProducer")

#--- initialize magnetic field B=3.8T
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# setting global tag
#process.GlobalTag.globaltag = "STARTUP31X_V5::All"
process.GlobalTag.globaltag = "STARTUP3XY_V9::All"



##__________________________________Filter____________________________________________
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(<FINEVT>)
)


process.load("Alignment.CommonAlignmentProducer.<ALCATAG>_Skimmed_cff")

process.pathALCARECOTkAlCosmicsCTFSkimmed = cms.Path(process.seqALCARECOTkAlCosmicsCTFSkimmed+process.NewStatsCTF)
#process.pathALCARECOTkAlCosmicsCosmicTFSkimmed = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFSkimmed )
#   path pathALCARECOTkAlCosmicsRSSkimmed =  { seqALCARECOTkAlCosmicsRSSkimmed }

##__________________________________Output____________________________________________

"""
process.OutALCARECOTkAlCosmicsSkimmed = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
         # Select as an OR of filters for the three tracking algorithms: 
 #   SelectEvents = cms.vstring( "pathALCARECOTkAlCosmicsCTFSkimmed","pathALCARECOTkAlCosmicsCosmicTFSkimmed")
    SelectEvents = cms.vstring( "pathALCARECOTkAlCosmicsCTFSkimmed")
    ),

    # replace "keep *_ALCARECOTkAlCosmics*Skimmed_*_*" with "keep *_ALCARECOTkAlCosmics*_*_*"
    # if you want to keep also former track collections
    outputCommands = cms.untracked.vstring( 
         "drop *" ,
        "keep *_ALCARECOTkAlCosmicsCTF0T*_*_*",
        "keep *_ALCARECOTkAlCosmicsCosmicTF0T*_*_*",
#         "keep *_cosmictrackfinderP5_*_*",
#         "keep *_ctfWithMaterialTracksP5_*_*",
         "keep *_*Skimmed*_*_*",
         "keep *_OverlapAssoMap*_*_*",
         "keep *_MEtoEDMConverter_*_*"
     ),
###    "keep Si*Cluster*_*_*_*", # for cosmics keep also clusters
    fileName = cms.untracked.string('file:./<JOB>_Skimmed.root'), 
    dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('ALCARECO'),
    filterName = cms.untracked.string('ALCARECOTkAlCosmicsSkimmed')
    )
   
 )
process.OutputALCARECOTkAlCosmicsSkimmed = cms.EndPath(process.OutALCARECOTkAlCosmicsSkimmed)
 """
##________end Output______________



###process.schedule = cms.Schedule(process.pathALCARECOTkAlCosmicsCTFSkimmed,process.pathALCARECOTkAlCosmicsCosmicTFSkimmed,process.OutputALCARECOTkAlCosmicsSkimmed)
#process.schedule = cms.Schedule(process.pathALCARECOTkAlCosmicsCTFSkimmed,process.OutputALCARECOTkAlCosmicsSkimmed)
process.schedule = cms.Schedule(process.pathALCARECOTkAlCosmicsCTFSkimmed)
