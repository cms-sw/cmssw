import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

#################################### Metadata ########################################
process = cms.Process("runMuonMillepedeAlgorithm")
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('runMuonMillepedeAlgorithm'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Alignment/MuonAlignmentAlgorithms/test/runDTLocalMillepedeFit.py,v $'),
    annotation = cms.untracked.string('runMuonMillepeAlgorithm')
)

###################################### Services #######################################
#MessageLogging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"))


#Databases 
process.load("CondCore.DBCommon.CondDBSetup_cfi")
#Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) 
#################################### Source block #####################################
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(nameOfTheFile)
    #fileNames = cms.untracked.vstring("/store/mc/Winter09/CosmicMC_BON_10GeV_AllCMS/ALCARECO/COSMMC_22X_V6_StreamALCARECOMuAlStandAloneCosmics_v1/0046/EA1AFD91-CB38-DE11-804E-003048322CA0.root")
)

#################################### Geometry ##########################################
process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
#process.load("Configuration.StandardSequences.MagneticField_38T_UpdatedMap_cff")
#process.VolumeBasedMagneticFieldESProducer.version = 'grid_1103l_090322_3_8t'
#process.load("MagneticField.Engine.volumeBasedMagneticField_1103l_090216_cfi")
process.load('Configuration/EventContent/EventContent_cff')

################################ Tags and databases ####################################
#process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
#process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
#     cms.PSet( record = cms.string("SiStripFedCablingRcd"), tag    = cms.string("") ),
#     cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
#     cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") )
#)
#process.prefer("SiStripQualityESProducer")


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRUZET4_V2P::All"
#process.GlobalTag.globaltag = "IDEAL_V11::All"
#process.GlobalTag.globaltag = "GR09_E_V2::All"
#process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"
#process.GlobalTag.globaltag = "IDEAL_V12::All"
process.GlobalTag.globaltag = "COSMMC_22X_V6::All"


del process.cscBadChambers
del process.DTFakeVDriftESProducer

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff") ## Needed at least on version 2_1_10

################################## Propagation ############################################
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")


process.looper.doMuon = cms.untracked.bool(True)
process.looper.applyDbAlignment = cms.untracked.bool(True)

process.load("Alignment.MuonAlignmentAlgorithms.MuonDTLocalMillepedeAlgorithmFit_cfi")
process.MuonDTLocalMillepedeAlgorithm.nMtxSection = cms.int32(5)

process.looper.algoConfig = cms.PSet(process.MuonDTLocalMillepedeAlgorithm)
process.looper.ParameterBuilder = cms.PSet(
  parameterTypes = cms.vstring('Selector,RigidBody4D'),
  Selector = cms.PSet(
   alignParams = cms.vstring('MuonDTChambers,110000')
  )
)

process.looper.tjTkAssociationMapTag = cms.InputTag("MuonMillepedeTrackRefitter")



#process.allPath = cms.Path(process.offlineBeamSpot * process.MuonAlignmentFromReferenceGlobalCosmicRefit)

################################# MuonStandaloneAlgorithm ####################################

#process.load("RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff")


process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("Alignment.MuonAlignmentAlgorithms.MuonMillepedeTrackRefitter_cfi")

process.allPath = cms.Path(process.offlineBeamSpot*process.MuonMillepedeTrackRefitter)



#For the new geometry 
#process.SiStripLorentzAngle = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(2),
#        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#    ),
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiStripLorentzAngleRcd'),
#       tag = cms.string('SiStripLA_CRAFT/GlobalAlignment_layers')
#    )),
#    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONALIGN/SWAlignment/Millepede/CRAFT/GlobalAlignment2/CMSSW_2_2_0/src/Alignment/MuonAlignmentAlgorithms/test/LA_CRAFT/GlobalAlignment_layers.db')#LA_CRAFT/GlobalAlignment_UNIFORM.db')
#)
#process.es_prefer_SiStripLorentzAngle = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngle")


#Muon Geometry
##import CondCore.DBCommon.CondDBSetup_cfi
##process.muonAlignment = cms.ESSource("PoolDBESSource",
##connect = cms.string("sqlite_file:/afs/cern.ch/user/s/scodella/scratch0/SoftwareAlignment/CMSSW_2_2_9/src/Alignment/MuonAlignment/test/MisalignmentScenarioForInternalAlignment.db"),
##DBParameters = CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup.DBParameters,
##toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"),       tag = cms.string("DTAlignmentRcd")),
##                  cms.PSet(record = cms.string("DTAlignmentErrorRcd"),  tag = cms.string("DTAlignmentErrorRcd")),
##                  cms.PSet(record = cms.string("CSCAlignmentRcd"),      tag = cms.string("CSCAlignmentRcd")),
##                  cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))
##process.es_prefer_muonAlignment = cms.ESPrefer("PoolDBESSource", "muonAlignment")




###TrackerGeometry
#process.trackerAlignment = cms.ESSource("PoolDBESSource",
#connect = cms.string("frontier://FrontierProd/CMS_COND_21X_ALIGNMENT"),
#connect = cms.string("sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONALIGN/SWAlignment/Millepede/CRAFT/GlobalAlignment/CMSSW_2_2_7/src/Alignment/MuonAlignmentAlgorithms/test/TrackerV4.db"),
#connect = cms.string("sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONALIGN/SWAlignment/Millepede/CRAFT/GlobalAlignment/CMSSW_2_2_10/src/Alignment/MuonAlignmentAlgorithms/test/TrackerAlignments.db"),
#DBParameters = CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup.DBParameters,
#toGet = cms.VPSet(
#cms.PSet(record = cms.string("TrackerAlignmentRcd"), tag = cms.string("Alignments")),
#cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"), tag = cms.string("AlignmentErrors"))
#))
#
#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
#process.looper.applyDbAlignment = cms.untracked.bool(True)



process.myRECO = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('reco.root') 
)


#process.outpath = cms.EndPath( process.myRECO )


