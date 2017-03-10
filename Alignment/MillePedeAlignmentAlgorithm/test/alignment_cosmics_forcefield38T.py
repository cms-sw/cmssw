# last update on $Date: 2008/08/11 19:12:41 $ by $Author: flucke $
#data name: cdc

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")

#-- MessageLogger
# process.load("FWCore.MessageLogger.MessageLogger_cfi")
# This whole mess does not really work - I do not get rid of FwkReport and TrackProducer info...
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring('alignment'), ##, 'cout')

    categories = cms.untracked.vstring('Alignment', 
        'LogicError', 
        'FwkReport', 
        'TrackProducer'),
    alignment = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('DEBUG'),
        LogicError = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        Alignment = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('alignment') ## (, 'cout')

)

#-- Magnetic field
#process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff") # for data
process.load("Configuration/StandardSequences/MagneticField_38T_cff") ## FOR 3.8T
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")  ## FOR 0T

##-----------------------------------------


#-- Ideal geometry and interface

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
# for Muon: process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
#process.load("Configuration.StandardSequences.Reconstruction_cff")#needed for the reconstruction in refit
#######################################
##Trigger settings for Cosmics during collisions
#######################################
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.L1T1=process.hltLevel1GTSeed.clone()
process.L1T1.L1TechTriggerSeeding = cms.bool(True)
process.L1T1.L1SeedsLogicalExpression=cms.string('25') 
process.hltHighLevel = cms.EDFilter("HLTHighLevel",
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    HLTPaths = cms.vstring('HLT_TrackerCosmics'),
    eventSetupPathsKey = cms.string(''),
    andOr = cms.bool(False),
    throw = cms.bool(True)
)


process.triggerSelection=cms.Sequence(process.L1T1*process.hltHighLevel)

## if alignment constants not from global tag, add this
from CondCore.DBCommon.CondDBSetup_cfi import *
#####################################################################
## Global Tag
######################################################################

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = "GR_P_V49::All" #"GR_P_V16::All" 
################################################################
## Load DBSetup (if needed)
####################################################################
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
##include private db object
##
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
###################################################################
## New Pixel LA
###################################################################

from CondCore.DBCommon.CondDBSetup_cfi import *

process.pixelTmpl = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string("SiPixelTemplateDBObjectRcd"),
            tag = cms.string("SiPixelTemplateDBObject_38T_2015_v1")
        )
    )
)
process.es_prefer_pixelTmpl = cms.ESPrefer("PoolDBESSource","pixelTmpl")

process.pixelLA = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string("SiPixelLorentzAngleRcd"),
        tag = cms.string("SiPixelLorentzAngle_2015_v2")
        )
    )
)
process.es_prefer_pixelLA = cms.ESPrefer("PoolDBESSource","pixelLA")




#placeholderstartgeometry


#placeholderLA


#-- initialize beam spot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

#-- track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'HitFilteredTracks'#'TrackRefitter2' # adjust to input file
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.pMin = 4
process.AlignmentTrackSelector.ptMin = 0
process.AlignmentTrackSelector.etaMin = -999.
process.AlignmentTrackSelector.etaMax = 999.
process.AlignmentTrackSelector.d0Min = -50.
process.AlignmentTrackSelector.d0Max = 50.
process.AlignmentTrackSelector.nHitMin = 8
#process.AlignmentTrackSelector.minHitsPerSubDet.inBPIX = 2
#process.AlignmentTrackSelector.minHitsPerSubDet.inTEC = 1
process.AlignmentTrackSelector.nHitMin2D = 2 
process.AlignmentTrackSelector.chi2nMax = 9999.
process.AlignmentTrackSelector.applyMultiplicityFilter = True# False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False # since no S/N cut is applied
process.AlignmentTrackSelector.minHitChargeStrip = 50.

# some further possibilities
#process.AlignmentTrackSelector.applyChargeCheck = True
#process.AlignmentTrackSelector.minHitChargeStrip = 50.
# needs RECO files:
#process.AlignmentTrackSelector.applyIsolationCut = True 
#process.AlignmentTrackSelector.minHitIsolation = 0.8


#-- new track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")#,"drop TID stereo","drop TEC stereo")


#placeholderdeadmodules



process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits = True
process.TrackerTrackHitFilter.TrackAngleCut = 0.35# in rads, starting from the module surface; .35 for cosmcics ok, .17 for collision tracks
process.TrackerTrackHitFilter.usePixelQualityFlag = True #False

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = 'TrackerTrackHitFilter',
	NavigationSchool = cms.string(''),
      TTRHBuilder = "WithAngleAndTemplate" #default
)

####################################################
#-- Alignment producer
####################################################
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")


#placeholderalignables
process.AlignmentProducer.checkDbAlignmentValidity = False

#process.AlignmentProducer.doMuon = True # to align muon system

process.AlignmentProducer.doMisalignmentScenario = True
# If the above is true, you might want to choose the scenario:
from Alignment.TrackerAlignment.Scenarios_cff import *
process.AlignmentProducer.MisalignmentScenario = TECRing7Minus133mmScenario

process.AlignmentProducer.applyDbAlignment = True  
process.AlignmentProducer.tjTkAssociationMapTag = "TrackRefitter2"

# monitor not strictly needed:
#process.TFileService = cms.Service("TFileService", fileName = cms.string("histograms.root"))
#process.AlignmentProducer.monitorConfig = cms.PSet(monitors = cms.untracked.vstring ("AlignmentMonitorGeneric"),
#                                                   AlignmentMonitorGeneric = cms.untracked.PSet()
#                                                   )

process.AlignmentProducer.algoConfig = cms.PSet(
    process.MillePedeAlignmentAlgorithm
)


#---Presigmas----------------
#from Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff import *
#process.AlignmentProducer.algoConfig.pedeSteerer.Presigmas += PresigmasCRAFT3rdMinBias.Presigmas
#--------------------------

process.AlignmentProducer.algoConfig.mode = 'mille'
process.AlignmentProducer.algoConfig.mergeBinaryFiles = cms.vstring()
process.AlignmentProducer.algoConfig.monitorFile = 'millePedeMonitorISN.root'
process.AlignmentProducer.algoConfig.binaryFile = 'milleBinaryISN.dat'
process.AlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
    # process.CombinedFwdBwdDualTrajectoryFactory
    # process.CombinedFwdBwdDualBzeroTrajectoryFactory
      process.BrokenLinesTrajectoryFactory
    #process.ReferenceTrajectoryFactory
)
process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = 'BrokenLinesCoarse' #Fine' #'BreakPoints'
process.AlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True # to account for multiple scattering in these layers

#placeholderpedesettings

#placeholderdetermineLA

## fwd propagation
##process.AlignmentProducer.algoConfig.TrajectoryFactory.Fwd.PropagationDirection = 'alongMomentum'
##bwd propagation
##process.AlignmentProducer.algoConfig.TrajectoryFactory.Bwd.PropagationDirection = 'oppositeToMomentum'

# refitting
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
    src ='ALCARECOTkAlCosmicsCTF0T',
    NavigationSchool = cms.string(''),
    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate" #default
    )

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'AlignmentTrackSelector',
#    TTRHBuilder = 'WithTrackAngle'
    )

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(
    '/store/AlcaReco/Commissioning08/Cosmics/ALCARECO/CRAFT0831X_V1_311_AlcaReco_FromSuperPointing_StreamTkAlCosmics0T_v1/0004/FCC2FE1B-6D74-DE11-8956-001A92971BB8.root'
# <== is a relval file from CMSSW_2_1_0_pre8.
        #"file:aFile.root" #"rfio:/castor/cern.ch/cms/store/..."
        )
)

##################################################################
##fix for cosmic tracks CPE error estimation--> fixed to 0.25
##################################################################
##process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
##process.StripCPEfromTrackAngleESProducer.CPEErrorCosmics=cms.untracked.bool(True)



process.dump = cms.EDAnalyzer("EventContentAnalyzer")

#process.p  = cms.Path(process.dump)


process.p = cms.Path( ###process.triggerSelection
                       process.offlineBeamSpot
                      *process.TrackRefitter1
                      *process.TrackerTrackHitFilter
                      *process.HitFilteredTracks
                      *process.AlignmentTrackSelector
                      *process.TrackRefitter2
                      )

#jsonfileplaceholder


#MILLEPEDEBLOCK
