######################################################################
######################################################################
otherTemplate = """
schum schum
"""


######################################################################
######################################################################
yResidualsOfflineValidation="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_Offline_.oO[name]Oo.', 
        'cout')
)

 ## report only every 100th record
 ##process.MessageLogger.cerr.FwkReport.reportEvery = 100

    
 ##
 ## Alignment Track Selection
 ##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'TrackRefitter1'
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

####  new FILTER
#-- new track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    369136710, 369136714, 402668822,
    # TOB
    436310989, 436310990, 436299301, 436299302,
    # TEC
    470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 14.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff.cosmictrackfinderP5.clone(
    src = 'TrackerTrackHitFilter'
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
  src = '.oO[TrackCollection]Oo.',
  TrajectoryInEvent = True,
  TTRHBuilder = "WithTrackAngle",
  NavigationSchool = ""
)

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'HitFilteredTracks')


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
#process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
# process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"

## LAYERWISE Lorentz Angle ###################

process.SiStripLorentzAngle = cms.ESSource("PoolDBESSource",
     BlobStreamerName = 
cms.untracked.string('TBufferBlobStreamingService'),
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(2),
         authenticationPath = 
cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
     ),
     timetype = cms.string('runnumber'),
     toGet = cms.VPSet(cms.PSet(
         record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLA_CRAFT_layers')
     )),
     connect = cms.string('sqlite_file:/afs/cern.ch/user/j/jdraeger/public/LA_object/LA_CRAFT_layers.db')
)
process.es_prefer_SiStripLorentzAngle = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngle")
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
process.load("Configuration/StandardSequences/MagneticField_38T_cff")

.oO[condLoad]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..stripYResiduals = True
.oO[offlineValidationFileOutput]Oo.

 ##
 ## PATH
 ##
process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.TrackRefitter2*process.AlignmentTrackSelector*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)

"""


######################################################################
######################################################################
zeroAPETemplate="""
from CondCore.DBCommon.CondDBSetup_cfi import *
process.APE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(
                                                          cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                                   tag = cms.string('TrackerIdealGeometryErrors210_mc')
                                                                   ))
                                        )
process.es_prefer_APE = cms.ESPrefer("PoolDBESSource", "APE")
"""


######################################################################
######################################################################
CosmicsOfflineValidation="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
#   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
reportEvery = cms.untracked.int32(1000) # every 1000th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
))
process.MessageLogger.statistics.append('cout')

 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

 #-- Refitting
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

##-- Track hit filter
## TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
#process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
#process.TrackerTrackHitFilter.src = 'TrackRefitter1'

#-- 1st refit from file
process.TrackRefitter1 = process.TrackRefitterP5.clone(
        src ='ALCARECOTkAlCosmicsCTF0T',
        NavigationSchool = cms.string(''),
        TrajectoryInEvent = True,
        TTRHBuilder = "WithAngleAndTemplate" #default
        )

#-- 2nd fit for AlignmentProducer
process.TrackRefitter2 = process.TrackRefitter1.clone(
       src = 'AlignmentTrackSelector'
       )
                            
#-- Filter bad hits
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = []
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits = True
process.TrackerTrackHitFilter.TrackAngleCut = 0.1# in rads, starting from the module surface; small value since we have bows!
process.TrackerTrackHitFilter.usePixelQualityFlag = True #False

#-- TrackProducer
## now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
process.TrackCandidateFitter = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
src = 'TrackerTrackHitFilter',
     NavigationSchool = cms.string(''),
     TTRHBuilder = "WithAngleAndTemplate"
     )
#-- Filter tracks for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'TrackCandidateFitter'
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.pMin = 4
process.AlignmentTrackSelector.pMax = 9999.
process.AlignmentTrackSelector.ptMin = 0
process.AlignmentTrackSelector.etaMin = -999.
process.AlignmentTrackSelector.etaMax = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 99.
process.AlignmentTrackSelector.applyMultiplicityFilter = True# False
process.AlignmentTrackSelector.maxMultiplicity = 1

## GlobalTag Conditions (if needed)
##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."


##
## Geometry
##
process.load("Configuration.StandardSequences.GeometryDB_cff")

##
## Magnetic Field
##
process.load("Configuration/StandardSequences/MagneticField_38T_cff")

import CalibTracker.Configuration.Common.PoolDBESSource_cfi

process.conditionsInTrackerAlignmentRcd = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1193/jobData/jobm/alignments_MP.db'),
          toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
          tag = cms.string('Alignments')
          )
          )
          )

process.prefer_conditionsInTrackerAlignmentRcd = cms.ESPrefer("PoolDBESSource", "conditionsInTrackerAlignmentRcd")

          
.oO[LorentzAngleTemplate]Oo.
  
 ##
 ## Geometry
 ##
#process.load("Configuration.StandardSequences.Geometry_cff")
 
.oO[condLoad]Oo.

 ##
 ## Load and Configure OfflineValidation
 ##

process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelProfiles = cms.bool(.oO[offlineModuleLevelProfiles]Oo.)
process.TFileService.fileName = '.oO[outputFile]Oo.'

 ##
 ## PATH
 ##

process.p = cms.Path( process.offlineBeamSpot
     *process.TrackRefitter1
     *process.TrackerTrackHitFilter
     *process.TrackCandidateFitter
     *process.AlignmentTrackSelector
     *process.TrackRefitter2
     *process.seqTrackerOfflineValidationStandalone
)


"""


######################################################################
######################################################################
CosmicsAt0TOfflineValidation="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
reportEvery = cms.untracked.int32(1000) # every 1000th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
))
process.MessageLogger.statistics.append('cout')


#-- Track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'

#-- Alignment Track Selection
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'HitFilteredTracks'
process.AlignmentTrackSelector.filter = True

.oO[TrackSelectionTemplate]Oo.
# Override the pmin setting since not meaningful with B=0T
process.AlignmentTrackSelector.pMin    = 4.

#### momentum constraint for 0T
# First momentum constraint
process.load("RecoTracker.TrackProducer.MomentumConstraintProducer_cff")
import RecoTracker.TrackProducer.MomentumConstraintProducer_cff
process.AliMomConstraint1 = RecoTracker.TrackProducer.MomentumConstraintProducer_cff.MyMomConstraint.clone()
process.AliMomConstraint1.src = '.oO[TrackCollection]Oo.'
process.AliMomConstraint1.fixedMomentum = 5.0
process.AliMomConstraint1.fixedMomentumError = 0.005

# Second momentum constraint
#process.load("RecoTracker.TrackProducer.MomentumConstraintProducer_cff")
#import RecoTracker.TrackProducer.MomentumConstraintProducer_cff
#process.AliMomConstraint2 = RecoTracker.TrackProducer.MomentumConstraintProducer_cff.MyMomConstraint.clone()
#process.AliMomConstraint2.src = 'AlignmentTrackSelector'
#process.AliMomConstraint2.fixedMomentum = 5.0
#process.AliMomConstraint2.fixedMomentumError = 0.005
   


#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = 'TrackerTrackHitFilter'
###    ,
###    TrajectoryInEvent = True,
###    TTRHBuilder = "WithAngleAndTemplate"    
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

#############
# parameters for TrackRefitter
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = process.TrackRefitterP5.clone(
   src =  '.oO[TrackCollection]Oo.', #'AliMomConstraint1',
   TrajectoryInEvent = True,
   TTRHBuilder = "WithAngleAndTemplate",
   NavigationSchool = "",
   constraint = 'momentum', ### SPECIFIC FOR CRUZET
   srcConstr='AliMomConstraint1' ### SPECIFIC FOR CRUZET$works only with tag V02-10-02 TrackingTools/PatternTools / or CMSSW >=31X
)

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'AlignmentTrackSelector',
    srcConstr='AliMomConstraint1',
    constraint = 'momentum' ### SPECIFIC FOR CRUZET
)


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
# process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"

.oO[LorentzAngleTemplate]Oo.
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
#process.load("Configuration/StandardSequences/MagneticField_38T_cff")
process.load("Configuration.StandardSequences.MagneticField_0T_cff") # 0T runs

.oO[condLoad]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)
process.TFileService.fileName = '.oO[outputFile]Oo.'

 ##
 ## PATH
 ##

process.p = cms.Path(process.offlineBeamSpot*process.AliMomConstraint1*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.AlignmentTrackSelector*process.TrackRefitter2*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)

"""


######################################################################
######################################################################
TrackSelectionCosmicsDuringCollisions = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

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
"""


######################################################################
######################################################################
TrackSelectionCosmicsData = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True
#process.TrackerTrackHitFilter.PxlCorrClusterChargeCut = 10000.0
#process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""


######################################################################
######################################################################
TrackSelectionCosmicsDataBPIX = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
process.AlignmentTrackSelector.minHitsPerSubDet = cms.PSet(
         inTEC = cms.int32(0),
         inTOB = cms.int32(0),
         inFPIX = cms.int32(0),
        inTID = cms.int32(0),
         inBPIX = cms.int32(1),
         inTIB = cms.int32(0),
         inPIXEL = cms.int32(0),
         inTIDplus = cms.int32(0),
         inTIDminus = cms.int32(0),
         inTECplus = cms.int32(0),
         inTECminus = cms.int32(0),
         inFPIXplus = cms.int32(0),
         inFPIXminus = cms.int32(0),
         inENDCAP = cms.int32(0),
         inENDCAPplus = cms.int32(0),
         inENDCAPminus = cms.int32(0),
     )
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""


######################################################################
######################################################################
TrackSelectionCosmicsDataFPIXplus = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
process.AlignmentTrackSelector.minHitsPerSubDet = cms.PSet(
         inTEC = cms.int32(0),
         inTOB = cms.int32(0),
         inFPIX = cms.int32(0),
        inTID = cms.int32(0),
         inBPIX = cms.int32(0),
         inTIB = cms.int32(0),
         inPIXEL = cms.int32(0),
         inTIDplus = cms.int32(0),
         inTIDminus = cms.int32(0),
         inTECplus = cms.int32(0),
         inTECminus = cms.int32(0),
         inFPIXplus = cms.int32(1),
         inFPIXminus = cms.int32(0),
         inENDCAP = cms.int32(0),
         inENDCAPplus = cms.int32(0),
         inENDCAPminus = cms.int32(0),
     )
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""


######################################################################
######################################################################
TrackSelectionCosmicsDataFPIXminus = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
process.AlignmentTrackSelector.minHitsPerSubDet = cms.PSet(
         inTEC = cms.int32(0),
         inTOB = cms.int32(0),
         inFPIX = cms.int32(0),
        inTID = cms.int32(0),
         inBPIX = cms.int32(0),
         inTIB = cms.int32(0),
         inPIXEL = cms.int32(0),
         inTIDplus = cms.int32(0),
         inTIDminus = cms.int32(0),
         inTECplus = cms.int32(0),
         inTECminus = cms.int32(0),
         inFPIXplus = cms.int32(0),
         inFPIXminus = cms.int32(1),
         inENDCAP = cms.int32(0),
         inENDCAPplus = cms.int32(0),
         inENDCAPminus = cms.int32(0),
     )
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""


######################################################################
######################################################################
TrackSelectionMinBiasData = """
##### For Tracks:collisions taken in deco mode
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 3
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.65
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,

    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.17 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

##############
##Trigger sequence
#############
#bit 0 is selecting bunch crossing
#bit 40 MinBias trigger


process.triggerSelection=cms.Sequence(process.oneGoodVertexFilter)

"""


######################################################################
######################################################################
TrackSelectionIsolatedMuons = """
##### For Tracks:collisions taken in deco mode

process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 0
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 3.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.17 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

##############
##Trigger sequence
#############
#bit 0 is selecting bunch crossing
#bit xy for muons trigger


#process.triggerSelection=cms.Sequence(process.bptxAnd)



"""


######################################################################
######################################################################
CosmicsOfflineValidation_Deformations="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

#process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
reportEvery = cms.untracked.int32(1000) # every 1000th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
))
process.MessageLogger.statistics.append('cout')


#-- Track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'



#-- Alignment Track Selection
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'HitFilteredTracks'
process.AlignmentTrackSelector.filter = True

.oO[TrackSelectionTemplate]Oo.

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = 'TrackerTrackHitFilter',
###    TrajectoryInEvent = True,
     NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate"    
)

 ##
 ## Load and Configure TrackRefitter1
 ##


#############
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = process.TrackRefitterP5.clone(
    src = 'ALCARECOTkAlCosmicsCTF0T',
    TrajectoryInEvent = True,
     NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate"
)
process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'AlignmentTrackSelector',
   
    )


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."


.oO[LorentzAngleTemplate]Oo.
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
process.load("Configuration/StandardSequences/MagneticField_38T_cff")

.oO[condLoad]Oo.

from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerSurfaces = cms.ESSource(
    "PoolDBESSource",
    CondDBSetup,
    connect = cms.string('.oO[dbpath]Oo.'),
    toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerSurfaceDeformationRcd"),
                              tag = cms.string("Deformations")
                               )
                      )
    )
process.es_prefer_trackerSurfaces = cms.ESPrefer("PoolDBESSource", "trackerSurfaces")

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##

process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
#process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)
process.TrackerOfflineValidationStandalone.moduleLevelHistsTransient = cms.bool(False)
process.TrackerOfflineValidationStandalone.moduleLevelProfiles    = cms.bool(True)
process.TFileService.fileName = '.oO[outputFile]Oo.'

 ##
 ## PATH
 ##
process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.AlignmentTrackSelector*process.TrackRefitter2*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)


"""


######################################################################
######################################################################
TrackSelectionCosmicsDataDef = """
# import CalibTracker.Configuration.Common.PoolDBESSource_cfi
# process.trackerBowedSensors = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
#      connect = cms.string('.oO[dbpath]Oo.'),
 
#     toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
#                                tag = cms.string('Deformations')
#                                )
#                       )
#     )
# process.prefer_trackerBowedSensors = cms.ESPrefer("PoolDBESSource", "trackerBowedSensors")
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = True
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True
#process.TrackerTrackHitFilter.PxlCorrClusterChargeCut = 10000.0
process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""


######################################################################
######################################################################
TrackSelectionCosmicsInterfillLADef = """
# import CalibTracker.Configuration.Common.PoolDBESSource_cfi
# process.trackerBowedSensors = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
#      connect = cms.string('.oO[dbpath]Oo.'),
 
#     toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
#                                tag = cms.string('Deformations')
#                                )
#                       )
#     )
# process.prefer_trackerBowedSensors = cms.ESPrefer("PoolDBESSource", "trackerBowedSensors")

#LA
process.myLA = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
        connect = cms.string ('frontier://PromptProd/CMS_COND_31X_STRIP'),
        toGet = cms.VPSet(cms.PSet(
                record = cms.string('SiStripLorentzAngleRcd'),
                tag = cms.string('SiStripLorentzAnglePeak_GR10_v1_offline')

                ))
        )
process.es_prefer_myLA = cms.ESPrefer("PoolDBESSource","myLA")

#-- initialize beam spot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

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
"""


######################################################################
######################################################################
TrackSelectionMinBiasDataDef = """
# import CalibTracker.Configuration.Common.PoolDBESSource_cfi
# process.trackerBowedSensors = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
#      connect = cms.string('.oO[dbpath]Oo.'),
 
#     toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
#                                tag = cms.string('Deformations')
#                                )
#                       )
#     )
# process.prefer_trackerBowedSensors = cms.ESPrefer("PoolDBESSource", "trackerBowedSensors")

##### For Tracks:collisions taken in deco mode
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 3
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.65
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,

    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.17 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

##############
##Trigger sequence
#############
#bit 0 is selecting bunch crossing
#bit 40 MinBias trigger


process.triggerSelection=cms.Sequence(process.oneGoodVertexFilter)

"""


######################################################################
######################################################################
TrackSelectionIsolatedMuonsDef = """
##### For Tracks:collisions taken in deco mode
# import CalibTracker.Configuration.Common.PoolDBESSource_cfi
# process.trackerBowedSensors = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
#      connect = cms.string('.oO[dbpath]Oo.'),
 
#     toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
#                                tag = cms.string('Deformations')
#                                )
#                       )
#     )
# process.prefer_trackerBowedSensors = cms.ESPrefer("PoolDBESSource", "trackerBowedSensors")


process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 3.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 5.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -3.
process.AlignmentTrackSelector.etaMax  = 3.
process.AlignmentTrackSelector.nHitMin = 10
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 30.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.17 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

##############
##Trigger sequence
#############
#bit 0 is selecting bunch crossing
#bit xy for muons trigger


#process.triggerSelection=cms.Sequence(process.bptxAnd)

"""
