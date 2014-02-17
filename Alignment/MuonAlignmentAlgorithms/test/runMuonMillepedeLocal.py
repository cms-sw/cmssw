######################################################################################
#################### runMuonStandaloneAlgorithm.py ###################################
###################################################################################### 

import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

#################################### Metadata ########################################
process = cms.Process("runMuonMillepedeAlgorithm")
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('runMuonMillepedeAlgorithm'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Alignment/MuonAlignmentAlgorithms/test/runMuonMillepedeLocal.py,v $'),
    annotation = cms.untracked.string('runMuonMillepeAlgorithm')
)

###################################### Services #######################################
#MessageLogging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO"))


#Databases 
process.load("CondCore.DBCommon.CondDBSetup_cfi")
#Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) 

#################################### Source block #####################################
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring("file:/tmp/pablom/0C0AB43B-D2A2-DD11-97EA-001617C3B6CC.root")
)


#################################### Geometry ##########################################
process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/EventContent/EventContent_cff')

################################ Tags and databases ####################################
#process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
#process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
#     cms.PSet( record = cms.string("SiStripFedCablingRcd"), tag    = cms.string("") ),
#     cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
#     cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") )
#)
#process.prefer("SiStripQualityESProducer")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRUZET4_V2P::All"
process.GlobalTag.globaltag = "IDEAL_V9::All"
process.prefer("GlobalTag")



process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")


################################## Propagation ############################################
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

# patch needed for CRUZET2 but not used in CRUZET3 cfg
#process.SteppingHelixPropagatorAny.useInTeslaFromMagField = True
#process.SteppingHelixPropagatorAlong.useInTeslaFromMagField = True
#process.SteppingHelixPropagatorOpposite.useInTeslaFromMagField = True
#process.SteppingHelixPropagatorAny.SetVBFPointer = True
#process.SteppingHelixPropagatorAlong.SetVBFPointer = True
#process.SteppingHelixPropagatorOpposite.SetVBFPointer = True
#process.SteppingHelixPropagator.SetVBFPointer = True


################################# MuonStandaloneAlgorithm ####################################
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
process.load("Alignment.MuonAlignmentAlgorithms.MuonMillepedeTrackRefitter_cfi")

process.MuonMillepedeTrackRefitter.MuonCollectionTag = cms.InputTag("ALCARECOMuAlGlobalCosmics:SelectedMuons")
process.MuonMillepedeTrackRefitter.TrackerTrackCollectionTag = cms.InputTag("ALCARECOMuAlGlobalCosmics:TrackerOnly")
process.MuonMillepedeTrackRefitter.SATrackCollectionTag = cms.InputTag("ALCARECOMuAlGlobalCosmics:StandAlone")


process.looper.doMuon = cms.untracked.bool(True)
process.looper.algoConfig = cms.PSet(process.MuonMillepedeAlgorithm)
process.looper.ParameterBuilder = cms.PSet(
  parameterTypes = cms.vstring('Selector,RigidBody4D'),
  Selector = cms.PSet(
    # selection of alignables and their parameters:
    # comma separated pairs of detector parts/levels as defined in AlinmentParameterSelector
    # (note special meaning if the string contains "SS" or "DS" or ends with "Layers"
    # followed by two digits)
    # and of d.o.f. to be aligned (x,y,z,alpha,beta,gamma) in local frame:
    # '0' means: deselect, '1' select. Others as 1, but might be interpreted in a special
    # way in the used algorithm (e.g. 'f' means fixed for millepede)
   alignParams = cms.vstring('MuonDTChambers,110000')
  )
)   

process.looper.tjTkAssociationMapTag = cms.InputTag("MuonMillepedeTrackRefitter")

#process.myRECO = cms.OutputModule("PoolOutputModule",
#     fileName = cms.untracked.string('reco.root')
#)


#Histogram services
process.TFileService = cms.Service("TFileService",
fileName = cms.string('Result.root')
)


process.MuonMillepedeAlgorithm.CollectionFile = cms.string('Resultado.root')
process.MuonMillepedeAlgorithm.isCollectionJob = cms.bool(False)
process.MuonMillepedeAlgorithm.collectionPath = cms.string("./job")
process.MuonMillepedeAlgorithm.collectionNumber = cms.int32(2)
process.MuonMillepedeAlgorithm.outputCollName = cms.string("FinalResult.root")
process.MuonMillepedeAlgorithm.ptCut = cms.double(10.0)
process.MuonMillepedeAlgorithm.chi2nCut = cms.double(6.0)

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.allPath = cms.Path( process.offlineBeamSpot * process.MuonMillepedeTrackRefitter )

#process.outpath = cms.EndPath( process.myRECO )


