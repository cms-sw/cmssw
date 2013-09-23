######################################################################
######################################################################
offlineTemplate = """
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
# process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")
.oO[datasetDefinition]Oo.

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

#  ##
#  ## Maximum number of Events
#  ## 
# process.maxEvents = cms.untracked.PSet(
#     input = cms.untracked.int32(.oO[nEvents]Oo.)
#  )

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

#############################################################
##select trigger bits 40 OR 41
##AND NOT (36 OR 37 OR 38 OR 39)
##trigger bit 0 is selecting crossing bunches (not in MC)
##########################################################
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

#Good Bunch Crossings
process.bptxAnd = process.hltLevel1GTSeed.clone(L1TechTriggerSeeding = cms.bool(True), L1SeedsLogicalExpression = cms.string('0'))
#BSCNOBEAMHALO
#process.bit40 = process.hltLevel1GTSeed.clone(L1TechTriggerSeeding = cms.bool(True), L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))'))

#Physics declared
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

    
#############################################################
##select only high purity tracks
##has to run first as necessary information
##is only available in initial track selection
##(Quality information is thrown away by the tracker refitters)
##########################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
process.HighPuritySelector = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    applyBasicCuts = True,
    filter = True,
    src = '.oO[TrackCollection]Oo.',
    trackQualities = ["highPurity"]
    )

###############################################################
## Quality filters on the event (REAL DATA - FIRST COLLISIONS DEC09 ONLY!)
## see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TRKPromptFeedBack#Event_and_track_selection_recipe
##NOT in PATH yet, to be commented in, if necessay
##
#########################################################    
############################################################################
##Produce Primary Vertex Collection needed for later analysis
############################################################################
#process.load('TrackingTools/TransientTrack/TransientTrackBuilder_cfi')
#process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
#process.offlinePrimaryVertices.TrackLabel = 'ALCARECOTkAlMinBias'

process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag("offlinePrimaryVertices"),
                                           cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
                                           filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
                                           )



process.FilterGoodEvents=cms.Sequence(#process.HighPuritySelector*
process.oneGoodVertexFilter)


process.noScraping= cms.EDFilter("FilterOutScraping",
                                 src=cms.InputTag("ALCARECOTkAlMinBias"),
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.25)
                                 )
####################################





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
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
   src = 'TrackerTrackHitFilter',
   NavigationSchool = "",

###    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate"    
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

#############
# parameters for TrackRefitter
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = process.TrackRefitter.clone(
   src = 'HighPuritySelector',
   TrajectoryInEvent = True,
   TTRHBuilder = "WithAngleAndTemplate",
   NavigationSchool = ""
)
process.TrackRefitter2 = process.TrackRefitter1.clone(
#    src = 'HitFilteredTracks')
     src = 'AlignmentTrackSelector'
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

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation and Output File
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = .oO[offlineModuleLevelHistsTransient]Oo.
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelProfiles = .oO[offlineModuleLevelProfiles]Oo.
.oO[offlineValidationFileOutput]Oo.

 ##
 ## PATH
 ##
process.p = cms.Path(
#process.triggerSelection*
process.offlineBeamSpot*process.HighPuritySelector*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
*process.AlignmentTrackSelector*process.TrackRefitter2*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)

"""


######################################################################
######################################################################
offlineParallelTemplate = """
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
 ## and number of events to be skipped
 ## in case of parallel job nIndex
 ## .oO[nIndex]Oo * .oO[nEvents]Oo/.oO[nJobs]Oo
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo./.oO[nJobs]Oo.)
 )
process.source.skipEvents=cms.untracked.uint32(.oO[nIndex]Oo.*.oO[nEvents]Oo./.oO[nJobs]Oo.)

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

#############################################################
##select trigger bits 40 OR 41
##AND NOT (36 OR 37 OR 38 OR 39)
##trigger bit 0 is selecting crossing bunches (not in MC)
##########################################################
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

#Good Bunch Crossings
process.bptxAnd = process.hltLevel1GTSeed.clone(L1TechTriggerSeeding = cms.bool(True), L1SeedsLogicalExpression = cms.string('0'))
#BSCNOBEAMHALO
#process.bit40 = process.hltLevel1GTSeed.clone(L1TechTriggerSeeding = cms.bool(True), L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))'))

#Physics declared
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

    
#############################################################
##select only high purity tracks
##has to run first as necessary information
##is only available in initial track selection
##(Quality information is thrown away by the tracker refitters)
##########################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
process.HighPuritySelector = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    applyBasicCuts = True,
    filter = True,
    src = '.oO[TrackCollection]Oo.',
    trackQualities = ["highPurity"]
    )

###############################################################
## Quality filters on the event (REAL DATA - FIRST COLLISIONS DEC09 ONLY!)
## see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TRKPromptFeedBack#Event_and_track_selection_recipe
##NOT in PATH yet, to be commented in, if necessay
##
#########################################################    
############################################################################
##Produce Primary Vertex Collection needed for later analysis
############################################################################
#process.load('TrackingTools/TransientTrack/TransientTrackBuilder_cfi')
#process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
#process.offlinePrimaryVertices.TrackLabel = 'ALCARECOTkAlMinBias'

process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag("offlinePrimaryVertices"),
                                           cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
                                           filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
                                           )



process.FilterGoodEvents=cms.Sequence(#process.HighPuritySelector*
process.oneGoodVertexFilter)


process.noScraping= cms.EDFilter("FilterOutScraping",
                                 src=cms.InputTag("ALCARECOTkAlMinBias"),
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.25)
                                 )
####################################





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
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
   src = 'TrackerTrackHitFilter',
   NavigationSchool = "",

###    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate"    
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

#############
# parameters for TrackRefitter
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = process.TrackRefitter.clone(
   src = 'HighPuritySelector',
   TrajectoryInEvent = True,
   TTRHBuilder = "WithAngleAndTemplate",
   NavigationSchool = ""
)
process.TrackRefitter2 = process.TrackRefitter1.clone(
#    src = 'HitFilteredTracks')
     src = 'AlignmentTrackSelector'
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

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation and Output File
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = .oO[offlineModuleLevelHistsTransient]Oo.
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelProfiles = .oO[offlineModuleLevelProfiles]Oo.
.oO[offlineValidationFileOutput]Oo.

 ##
 ## PATH
 ##
process.p = cms.Path(
#process.triggerSelection*
process.offlineBeamSpot*process.HighPuritySelector*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
*process.AlignmentTrackSelector*process.TrackRefitter2*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)

"""


######################################################################
######################################################################
mergeOfflineParallelResults="""

# Merging works also if there is only one file to merge
# if merged file already exists it will be moved to a backup file (~)

# run TkAlOfflinejobs.C
echo "Merging results from parallel jobs with TkAlOfflineJobsMerge.C"
#set directory to which TkAlOfflineJobsMerge.C saves the merged file
# export OUTPUTDIR=.oO[datadir]Oo.
export OUTPUTDIR=.
cp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/merge_TrackerOfflineValidation.C .
.oO[haddLoop]Oo.

# create log file
# ls -al .oO[datadir]Oo./AlignmentValidation*.root > .oO[datadir]Oo./log_rootfilelist.txt
ls -al AlignmentValidation*.root > .oO[datadir]Oo./log_rootfilelist.txt

# Remove parallel job files if merged file exists
for file in $(cmsLs -l /store/caf/user/$USER/.oO[eosdir]Oo. |awk '{print $5}')
do
    if [[ ${file} = /store/caf/user/$USER/.oO[eosdir]Oo./AlignmentValidation*_[0-9].root ]]
    then
        cmsRm ${file}
    fi
done

"""


######################################################################
######################################################################
mergeOfflineParJobsTemplate="""
void TkAlOfflineJobsMerge(TString pars, TString outFile)
{
// load framework lite just to find the CMSSW libs...
gSystem->Load("libFWCoreFWLite");
AutoLibraryLoader::enable();
//compile the macro
gROOT->ProcessLine(".L merge_TrackerOfflineValidation.C++");

hadd(pars, outFile);
}
"""


######################################################################
######################################################################
offlineStandaloneFileOutputTemplate = """
process.TFileService.fileName = '.oO[outputFile]Oo.'
"""

######################################################################
######################################################################
offlineParallelFileOutputTemplate = """
process.TFileService.fileName = '.oO[outputFile]Oo.'
"""

######################################################################
######################################################################
offlineDqmFileOutputTemplate = """
process.TrackerOfflineValidationSummary.oO[offlineValidationMode]Oo..removeModuleLevelHists = .oO[offlineModuleLevelHistsTransient]Oo.
process.DqmSaverTkAl.workflow = '.oO[workflow]Oo.'
process.DqmSaverTkAl.dirName = '.oO[workdir]Oo./.'
process.DqmSaverTkAl.forceRunNumber = .oO[firstRunNumber]Oo.
"""
#offlineDqmFileOutputTemplate = """
#process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..workflow =  .oO[workflow]Oo.
#process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..dirName = .oO[workdir]Oo./.
#process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..forceRunNumber = .oO[firstRunNumber]Oo.
#"""


######################################################################
######################################################################
LorentzAngleTemplate = "#use lorentz angle from global tag"


######################################################################
######################################################################
TrackSelectionTemplate = """
#####default for MC tracks with now further corrections etc.

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
#for MC only trigger bit 40 is simulated
#no triger on bunch crossing bit 0


# process.triggerSelection=cms.Sequence(process.bit40)

"""

