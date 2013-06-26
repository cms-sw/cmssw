import FWCore.ParameterSet.Config as cms

process = cms.Process("ALCAPRESCTkAl")

##__________________________Messages & Convenience____________________________________-
process.load("FWCore.MessageService.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
       destinations = cms.untracked.vstring('LOGFILE_CosmicsPrescaled_<JOB>'),
       statistics   = cms.untracked.vstring('LOGFILE_CosmicsPrescaled_<JOB>'),
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
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# setting global tag
process.GlobalTag.globaltag = "STARTUP3XY_V9::All"

##__________________________________Filter____________________________________________
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(<FINEVT>)
)


process.load("Alignment.CommonAlignmentProducer.<ALCATAG>_Skimmed_cff")

process.load("Alignment.TrackerAlignment.AlignmentPrescaler_cff")
process.TkAlHitAssoMapCTF =  process.AlignmentPrescaler.clone()
process.TkAlHitAssoMapCTF.src="ALCARECOTkAlCosmicsCTF4TSkimmed"
process.TkAlHitAssoMapCTF.assomap="OverlapAssoMapCTF"
process.TkAlHitAssoMapCTF.PrescFileName="<MERGEDHITMAP>"#same as process.AlignmentTreeMerger.OutputFile


process.pathALCARECOTkAlCosmicsCTFPrescaled = cms.Path(process.seqALCARECOTkAlCosmicsCTFSkimmed+process.TkAlHitAssoMapCTF)


##__________________________________Output____________________________________________


process.OutALCARECOTkAlCosmicsPrescaled = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
         # Select as an OR of filters for the three tracking algorithms: 
    SelectEvents = cms.vstring( "pathALCARECOTkAlCosmicsCTFPrescaled")
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
         "keep *_TkAlHitAssoMap*_*_*",
       'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
         "keep *_MEtoEDMConverter_*_*"
     ),
###    "keep Si*Cluster*_*_*_*", # for cosmics keep also clusters
    fileName = cms.untracked.string('file:./<JOB>_Prescaled.root'), 
    dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('ALCARECO'),
    filterName = cms.untracked.string('ALCARECOTkAlCosmicsPrescaled')
    )
   
 )


process.OutputALCARECOTkAlCosmicsPrescaled = cms.EndPath(process.OutALCARECOTkAlCosmicsPrescaled)
process.schedule = cms.Schedule(process.pathALCARECOTkAlCosmicsCTFPrescaled,process.OutputALCARECOTkAlCosmicsPrescaled)
