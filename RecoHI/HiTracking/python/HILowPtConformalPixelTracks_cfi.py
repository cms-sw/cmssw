import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cfi import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

from RecoPixelVertexing.PixelTrackFitting.PixelFitterByConformalMappingAndLine_cfi import *

from RecoHI.HiTracking.hiMultiTrackSelector_cfi import *
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import *

hiConformalPixelTracks = cms.EDProducer("PixelTrackProducer",
                                        
                                        #passLabel  = cms.string('Pixel triplet low-pt tracks with vertex constraint'),
                                        
                                        # Region
                                        RegionFactoryPSet = cms.PSet(
    ComponentName = cms.string("GlobalTrackingRegionWithVerticesProducer"),
    RegionPSet = cms.PSet(
    HiLowPtTrackingRegionWithVertexBlock
    )
    ),
                                        
                                        # Ordered Hits
                                        OrderedHitsFactoryPSet = cms.PSet( 
    ComponentName = cms.string( "StandardHitTripletGenerator" ),
    SeedingLayers = cms.InputTag( "PixelLayerTriplets" ),
    GeneratorPSet = cms.PSet( 
    PixelTripletHLTGenerator
    )
    ),
                                        
                                        # Fitter
                                        FitterPSet = cms.PSet( PixelFitterByConformalMappingAndLine 
                                                               ),
                                        
                                        # Filter
                                        useFilterWithES = cms.bool( True ),
                                        FilterPSet = cms.PSet( 
    HiConformalPixelFilterBlock 
    ),
                                        
                                        # Cleaner
                                        CleanerPSet = cms.PSet(  
    ComponentName = cms.string( "TrackCleaner" )
    )
                                        )

# increase threshold for triplets in generation step (default: 10000)
hiConformalPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 5000000

hiConformalPixelTracks.FitterPSet.fixImpactParameter = cms.double(0.0)
hiConformalPixelTracks.FitterPSet.TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')

# Selector for quality pixel tracks with tapering high-pT cut

#loose
hiPixelOnlyStepLooseMTS = hiLooseMTS.clone(
    name= cms.string('hiPixelOnlyTrkLoose'),
    chi2n_no1Dmod_par = cms.double(25.0),
    d0_par2 = cms.vdouble(9999.0, 0.0),              # d0E from tk.d0Error
    dz_par2 = cms.vdouble(14.0, 0.0), 
    max_relpterr = cms.double(9999.),
    min_nhits = cms.uint32(0),
    applyHIonCuts = cms.bool(True),
    hIon_pTMaxCut = cms.vdouble(10,5,25,2.5)
)

hiPixelOnlyStepTightMTS=hiPixelOnlyStepLooseMTS.clone(
    preFilterName='hiPixelOnlyTrkLoose',
    chi2n_no1Dmod_par = cms.double(18.0),
    dz_par2 = cms.vdouble(12.0, 0.0),
    hIon_pTMaxCut = cms.vdouble(4,2,18,2.5),
    name= cms.string('hiPixelOnlyTrkTight'),
    qualityBit = cms.string('tight'),
    keepAllTracks= cms.bool(True)
    )

hiPixelOnlyStepHighpurityMTS= hiPixelOnlyStepTightMTS.clone(
    name= cms.string('hiPixelOnlyTrkHighPurity'),
    preFilterName='hiPixelOnlyTrkTight',
    chi2n_no1Dmod_par = cms.double(12.),    
    dz_par2 = cms.vdouble(10.0, 0.0),
    hIon_pTMaxCut = cms.vdouble(2.4,1.6,12,2.5),
    qualityBit = cms.string('highPurity') ## set to '' or comment out if you dont want to set the bit
    )

hiPixelOnlyStepSelector = hiMultiTrackSelector.clone(
    src='hiConformalPixelTracks',
    trackSelectors= cms.VPSet(
        hiPixelOnlyStepLooseMTS,
        hiPixelOnlyStepTightMTS,
        hiPixelOnlyStepHighpurityMTS
    ) #end of vpset
    ) #end of clone


# selector for tapered full tracks

hiHighPtStepTruncMTS = hiLooseMTS.clone(
    name= cms.string('hiHighPtTrkTrunc'),
    chi2n_no1Dmod_par = cms.double(9999.0),
    d0_par2 = cms.vdouble(9999.0, 0.0),              # d0E from tk.d0Error
    dz_par2 = cms.vdouble(9999.0, 0.0),
    max_relpterr = cms.double(9999.),
    min_nhits = cms.uint32(12),
    applyHIonCuts = cms.bool(True),
    hIon_pTMinCut = cms.vdouble(1.0,1.8,0.15,2.5),
    qualityBit = cms.string('')
)

hiHighPtStepSelector = hiMultiTrackSelector.clone(
    src='hiGeneralTracks',
    trackSelectors= cms.VPSet(
        hiHighPtStepTruncMTS
    ) #end of vpset
    ) #end of clone


# make final collection, unmerged for now

hiGeneralAndPixelTracks = trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiConformalPixelTracks'),
                          cms.InputTag('hiGeneralTracks')
                     ),
    hasSelector=cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(
    cms.InputTag("hiPixelOnlyStepSelector","hiPixelOnlyTrkHighPurity"),
    cms.InputTag("hiHighPtStepSelector","hiHighPtTrkTrunc")
#    cms.InputTag('')
    ),                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(False)), 
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
