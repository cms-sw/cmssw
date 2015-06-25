import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForSiPixelAli = copy.deepcopy(hltHighLevel)
ALCARECOTkAlMinBiasFilterForSiPixelAli.HLTPaths = ['pathALCARECOTkAlMinBias']
ALCARECOTkAlMinBiasFilterForSiPixelAli.throw = True ## dont throw on unknown path names
ALCARECOTkAlMinBiasFilterForSiPixelAli.TriggerResultsTag = cms.InputTag("TriggerResults","","reRECO")




# Ingredient: offlineBeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import offlineBeamSpot

# Ingredient: AlignmentTrackSelector
# track selection for alignment
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import AlignmentTrackSelector
AlignmentTrackSelector.src = 'ALCARECOTkAlMinBias'  #'SiPixelAliTrackFitter' #'ALCARECOTkAlCosmicsCTF0T' #TkAlZMuMu' #MinBias' #'generalTracks' ## ALCARECOTkAlMuonIsolated # adjust to input file
AlignmentTrackSelector.pMin = 4.
AlignmentTrackSelector.ptMin = 0. #HIGHER CUT, LESS TRACKS, MORE EVENTS, LESS TIME THOUGH?????
AlignmentTrackSelector.ptMax = 200.
AlignmentTrackSelector.etaMin = -999.
AlignmentTrackSelector.etaMax = 999.
AlignmentTrackSelector.nHitMin = 10
AlignmentTrackSelector.nHitMin2D = 3
AlignmentTrackSelector.chi2nMax = 100.
AlignmentTrackSelector.applyMultiplicityFilter = False# True
AlignmentTrackSelector.maxMultiplicity = 1
AlignmentTrackSelector.minHitsPerSubDet.inPIXEL = 2


# Ingredient: SiPixelAliTrackRefitter0
# refitting
from RecoTracker.TrackProducer.TrackRefitters_cff import *
# In the following use
# TrackRefitter (normal tracks), TrackRefitterP5 (cosmics) or TrackRefitterBHM (beam halo)

SiPixelAliTrackRefitter0 = TrackRefitter.clone(
        src = 'AlignmentTrackSelector',   #'ALCARECOTkAlMinBias'#'ALCARECOTkAlCosmicsCTF0T' #'ALCARECOTkAlMuonIsolated'
        NavigationSchool = '',            # to avoid filling hit pattern
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

SiPixelAliMilleAlignmentProducer.MisalignmentScenario = cms.PSet(
        setRotations = cms.bool(True),
        setTranslations = cms.bool(True),
        seed = cms.int32(1234567),
        distribution = cms.string('fixed'), # gaussian, uniform (or so...)
        setError = cms.bool(True), #GF ???????
        TPBHalfBarrel1 = cms.PSet(
                dXlocal = cms.double(0.0020),
                dYlocal = cms.double(-0.0015),
                dZlocal = cms.double(0.0100),
                phiXlocal = cms.double(1.e-4),
                phiYlocal = cms.double(-2.e-4),
                phiZlocal = cms.double(5.e-4),

                ),
        TPBHalfBarrel2 = cms.PSet(
                dXlocal = cms.double(-0.0020),
                dYlocal = cms.double(0.0030),
                dZlocal = cms.double(-0.020),
                phiXlocal = cms.double(1.e-3),
                phiYlocal = cms.double(2.e-4),
                phiZlocal = cms.double(-2.e-4),

                ),
        TPEEndcap1 = cms.PSet(
                TPEHalfCylinder1 = cms.PSet(
                        dXlocal = cms.double(0.0050),
                        dYlocal = cms.double(0.0020),
                        dZlocal = cms.double(-0.005),
                        phiXlocal = cms.double(-1.e-5),
                        phiYlocal = cms.double(2.e-3),
                        phiZlocal = cms.double(2.e-5),
                    ),
                TPEHalfCylinder2 = cms.PSet(
                        dXlocal = cms.double(0.0020),
                        dYlocal = cms.double(0.0030),
                        dZlocal = cms.double(-0.01),
                        phiXlocal = cms.double(1.e-4),
                        phiYlocal = cms.double(-1.e-4),
                        phiZlocal = cms.double(2.e-4),
                    ),
            ),
        TPEEndcap2 = cms.PSet(
                TPEHalfCylinder1 = cms.PSet(
                        dXlocal = cms.double(-0.0080),
                        dYlocal = cms.double(0.0050),
                        dZlocal = cms.double(-0.005),
                        phiXlocal = cms.double(1.e-3),
                        phiYlocal = cms.double(-3.e-4),
                        phiZlocal = cms.double(2.e-4),
                    ),
                TPEHalfCylinder2 = cms.PSet(
                        dXlocal = cms.double(0.0020),
                        dYlocal = cms.double(0.0030),
                        dZlocal = cms.double(-0.005),
                        phiXlocal = cms.double(-1.e-3),
                        phiYlocal = cms.double(2.e-4),
                        phiZlocal = cms.double(3.e-4),
                    ),
            )
    )

SiPixelAliMilleAlignmentProducer.checkDbAlignmentValidity = False
SiPixelAliMilleAlignmentProducer.applyDbAlignment = True
SiPixelAliMilleAlignmentProducer.tjTkAssociationMapTag = 'SiPixelAliTrackFitter'

SiPixelAliMilleAlignmentProducer.algoConfig = MillePedeAlignmentAlgorithm
SiPixelAliMilleAlignmentProducer.algoConfig.mode = 'mille'
SiPixelAliMilleAlignmentProducer.algoConfig.mergeBinaryFiles = cms.vstring()
SiPixelAliMilleAlignmentProducer.algoConfig.binaryFile = 'milleBinary0.dat'
SiPixelAliMilleAlignmentProducer.algoConfig.TrajectoryFactory = BrokenLinesTrajectoryFactory
#SiPixelAliMilleAlignmentProducer.algoConfig.TrajectoryFactory.MomentumEstimate = 10
SiPixelAliMilleAlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = 'BrokenLinesCoarse' #Coarse' #Fine' #'BreakPoints'
SiPixelAliMilleAlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True # to account for multiple scattering in these layers




# Ingredient: SiPixelAliTrackerTrackHitFilter
import RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff as HitFilter
# Reference config at /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/aliConfigTemplates/Cosmics38T_BL_default_cfg.py
SiPixelAliTrackerTrackHitFilter = HitFilter.TrackerTrackHitFilter.clone(
    src = 'SiPixelAliTrackRefitter0', #'ALCARECOTkAlCosmicsCTF0T',
    useTrajectories= False,#True, # for angle selections + pixel cluster charge
    minimumHits = 8,
    commands = [], # Ref. has equivalent pharse...
    detsToIgnore = [], #is default
    replaceWithInactiveHits = True, # needed for multiple scattering
    stripAllInvalidHits = False, #default
    rejectBadStoNHits = True,
    StoNcommands = ["ALL 18.0"], # 18 for tracker in peak mode, 5 for deconvolution mode
#    rejectLowAngleHits = True,
    TrackAngleCut = 0.1, # 0.35, # in rads, starting from the module surface; .35 for cosmcics ok, .17 for collision tracks
    usePixelQualityFlag = True
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

seqALCARECOPromptCalibProdSiPixelAli = cms.Sequence(ALCARECOTkAlMinBiasFilterForSiPixelAli*
                                                    offlineBeamSpot*
                                                    AlignmentTrackSelector*
                                                    SiPixelAliTrackRefitter0*
                                                    SiPixelAliTrackerTrackHitFilter*
                                                    SiPixelAliTrackFitter*
                                                    SiPixelAliMilleAlignmentProducer)
                                                    ##SiPixelAliMillePedeFileConverter)
