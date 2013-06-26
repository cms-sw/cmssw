import FWCore.ParameterSet.Config as cms

process = cms.Process("ALCASKIMTkAl")


##__________________________Messages & Convenience____________________________________-
process.load("FWCore.MessageService.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
       destinations = cms.untracked.vstring('LOGFILE_MinBiasSkimmed_<JOB>'),
       statistics   = cms.untracked.vstring('LOGFILE_MinBiasSkimmed_<JOB>'),
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
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

##__________________________________Source_____________________________________________
process.source = cms.Source("PoolSource",
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
#process.GlobalTag.globaltag = "GR_R_38X_V13A::All"
process.GlobalTag.globaltag = "GR10_P_V11::All"


##__________________________________Filter____________________________________________
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(<FINEVT>)
)


process.load("Alignment.CommonAlignmentProducer.<ALCATAG>_Skimmed_cff")

process.pathALCARECOTkAlMinBiasSkimmed = cms.Path(process.seqALCARECOTkAlMinBiasSkimmed+process.NewStatsCTF)
#process.pathALCARECOTkAlCosmicsCosmicTFSkimmed = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFSkimmed )
#   path pathALCARECOTkAlCosmicsRSSkimmed =  { seqALCARECOTkAlCosmicsRSSkimmed }

##__________________________________Output____________________________________________

"""
process.OutALCARECOTkAlMinBiasSkimmed = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
         # Select as an OR of filters for the three tracking algorithms: 
 #   SelectEvents = cms.vstring( "pathALCARECOTkAlMinBiasCTFSkimmed","pathALCARECOTkAlMinBiasCosmicTFSkimmed")
    SelectEvents = cms.vstring( "pathALCARECOTkAlMinBiasSkimmed")
    ),

    # replace "keep *_ALCARECOTkAlMinBias*Skimmed_*_*" with "keep *_ALCARECOTkAlMinBias*_*_*"
    # if you want to keep also former track collections
    outputCommands = cms.untracked.vstring( 
         "drop *" ,
        "keep *_ALCARECOTkAlMinBias*_*_*",
        "keep *_ALCARECOTkAlMinBiasRS*_*_*",
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
    filterName = cms.untracked.string('ALCARECOTkAlMinBiasSkimmed')
    )
   
 )
process.OutputALCARECOTkAlMinBiasSkimmed = cms.EndPath(process.OutALCARECOTkAlMinBiasSkimmed)
 """
##________end Output______________



###process.schedule = cms.Schedule(process.pathALCARECOTkAlMinBiasCTFSkimmed,process.pathALCARECOTkAlMinBiasCosmicTFSkimmed,process.OutputALCARECOTkAlMinBiasSkimmed)
#process.schedule = cms.Schedule(process.pathALCARECOTkAlMinBiasSkimmed,process.OutputALCARECOTkAlMinBiasSkimmed)
process.schedule = cms.Schedule(process.pathALCARECOTkAlMinBiasSkimmed)
