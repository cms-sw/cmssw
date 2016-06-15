import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForSiPixelAli = copy.deepcopy(hltHighLevel)
ALCARECOTkAlMinBiasFilterForSiPixelAli.HLTPaths = ['pathALCARECOTkAlMinBias']
ALCARECOTkAlMinBiasFilterForSiPixelAli.throw = True ## dont throw on unknown path names
ALCARECOTkAlMinBiasFilterForSiPixelAli.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")

from Alignment.CommonAlignmentProducer.LSNumberFilter_cfi import *


# Ingredient: offlineBeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import offlineBeamSpot

# Ingredient: AlignmentTrackSelector
# track selector for HighPurity tracks
#-- AlignmentTrackSelector
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import AlignmentTrackSelector
SiPixelAliHighPuritySelector = AlignmentTrackSelector.clone(
        applyBasicCuts = True,
        #filter = True,
        src = 'ALCARECOTkAlMinBias',
        trackQualities = ["highPurity"],
		pMin = 4.9, #for 0T Collisions
		pMax = 5.1, #for 0T Collisions
        )



# track selection for alignment
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import AlignmentTrackSelector
SiPixelAliTrackSelector = AlignmentTrackSelector.clone(
	src = 'SiPixelAliTrackFitter',
	applyBasicCuts = True,
	pMin = 4.9,  #for 0T Collisions                                                                            
	pMax = 5.1, #for 0T Collisions
	ptMin = 0., #for 0T Collisions
	etaMin = -999.,
	etaMax = 999.,
	nHitMin = 8,
	nHitMin2D = 2,
	chi2nMax = 9999.,
	applyMultiplicityFilter = False,
	maxMultiplicity = 1,
	applyNHighestPt = False,
	nHighestPt = 1,
	seedOnlyFrom = 0,
	applyIsolationCut = False,
	minHitIsolation = 0.8,
	applyChargeCheck = False,
	minHitChargeStrip = 30.,
)
#Special option for PCL
SiPixelAliTrackSelector.minHitsPerSubDet.inPIXEL = 2


# Ingredient: SiPixelAliTrackRefitter0
# refitting
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *
# In the following use
# TrackRefitter (normal tracks), TrackRefitterP5 (cosmics) or TrackRefitterBHM (beam halo)

SiPixelAliTrackRefitter0 = TrackRefitter.clone(
        src = 'SiPixelAliHighPuritySelector',   #'ALCARECOTkAlMinBias'#'ALCARECOTkAlCosmicsCTF0T' #'ALCARECOTkAlMuonIsolated'
        NavigationSchool = '',            # to avoid filling hit pattern
                                              )

SiPixelAliTrackRefitter1 = SiPixelAliTrackRefitter0.clone(
	src = 'SiPixelAliTrackSelector'
)

#-- Alignment producer
from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.CommonAlignmentProducer.TrackerAlignmentProducerForPCL_cff import AlignmentProducer 
SiPixelAliMilleAlignmentProducer = copy.deepcopy(AlignmentProducer)
SiPixelAliMilleAlignmentProducer.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        'TrackerTPBHalfBarrel,111111',
        'TrackerTPEHalfCylinder,111111',

        'TrackerTIBHalfBarrel,ffffff',
        'TrackerTOBHalfBarrel,ffffff',
        'TrackerTIDEndcap,ffffff',
        'TrackerTECEndcap,ffffff'
        )
    )

SiPixelAliMilleAlignmentProducer.doMisalignmentScenario = False #True


SiPixelAliMilleAlignmentProducer.checkDbAlignmentValidity = False
SiPixelAliMilleAlignmentProducer.applyDbAlignment = True
SiPixelAliMilleAlignmentProducer.tjTkAssociationMapTag = 'SiPixelAliTrackRefitter1'

SiPixelAliMilleAlignmentProducer.algoConfig = MillePedeAlignmentAlgorithm
SiPixelAliMilleAlignmentProducer.algoConfig.mode = 'mille'
SiPixelAliMilleAlignmentProducer.algoConfig.runAtPCL = True
SiPixelAliMilleAlignmentProducer.algoConfig.mergeBinaryFiles = cms.vstring()
SiPixelAliMilleAlignmentProducer.algoConfig.binaryFile = 'milleBinary_0.dat'
SiPixelAliMilleAlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
      BrokenLinesBzeroTrajectoryFactory # For 0T collisions
      )
SiPixelAliMilleAlignmentProducer.algoConfig.TrajectoryFactory.MomentumEstimate = 5 #for 0T Collisions



# Ingredient: SiPixelAliTrackerTrackHitFilter
import RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff as HitFilter
# Reference config at /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/aliConfigTemplates/Cosmics38T_BL_default_cfg.py
SiPixelAliTrackerTrackHitFilter = HitFilter.TrackerTrackHitFilter.clone(
    src = 'SiPixelAliTrackRefitter0', #'ALCARECOTkAlCosmicsCTF0T',	
    # this is needed only if you require some selections; but it will work even if you don't ask for them
    useTrajectories= True,
    minimumHits = 8,
    replaceWithInactiveHits = True,
    rejectBadStoNHits = True,
    commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC"), #,"drop TID stereo","drop TEC stereo")
    stripAllInvalidHits = False,
    StoNcommands = cms.vstring("ALL 12.0"),
    rejectLowAngleHits = True,
    TrackAngleCut = 0.17, # in rads, starting from the module surface; .35 for cosmcics ok, .17 for collision tracks
    usePixelQualityFlag = True #False
    )

# Ingredient: SiPixelAliSiPixelAliTrackFitter
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff as fitWithMaterial
SiPixelAliTrackFitter = fitWithMaterial.ctfWithMaterialTracks.clone(
        src = 'SiPixelAliTrackerTrackHitFilter',
        # TTRHBuilder = 'WithAngleAndTemplate', #should already be default
        NavigationSchool = ''
        )

### Ingredient: MillePedeFileConverter
##from Alignment.CommonAlignmentProducer.MillePedeFileConverter_cfi import millePedeFileConverter
### We configure the input file name of the millePedeFileConverter
###         with the output file name of the alignmentProducer (=looper).
### Like this we are sure that they are well connected.
##SiPixelAliMillePedeFileConverter = millePedeFileConverter.clone(
##        fileDir = looper.algoConfig.fileDir,
##        binaryFile = looper.algoConfig.binaryFile,
##        )

SiPixelAliMillePedeFileConverter = cms.EDProducer("MillePedeFileConverter",
                                                  #FIXME: convert to untracked?
                                                  fileDir = cms.string(SiPixelAliMilleAlignmentProducer.algoConfig.fileDir.value()),
                                                  inputBinaryFile = cms.string(SiPixelAliMilleAlignmentProducer.algoConfig.binaryFile.value()),
                                                  #FIXME: why was the label removed? Don't we want a label?
                                                  fileBlobLabel = cms.string(''),
                                                 )



seqALCARECOPromptCalibProdSiPixelAli = cms.Sequence(ALCARECOTkAlMinBiasFilterForSiPixelAli*
                                                    lsNumberFilter*
                                                    offlineBeamSpot*
                                                    SiPixelAliHighPuritySelector*
                                                    SiPixelAliTrackRefitter0*
                                                    SiPixelAliTrackerTrackHitFilter*
                                                    SiPixelAliTrackFitter*
						    SiPixelAliTrackSelector*
						    SiPixelAliTrackRefitter1*
                                                    SiPixelAliMilleAlignmentProducer*
                                                    SiPixelAliMillePedeFileConverter)
