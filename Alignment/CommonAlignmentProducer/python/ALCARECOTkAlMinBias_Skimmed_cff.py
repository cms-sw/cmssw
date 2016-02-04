import FWCore.ParameterSet.Config as cms

#from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
import RecoVertex.BeamSpotProducer.BeamSpot_cfi
MyBeamSpot= RecoVertex.BeamSpotProducer.BeamSpot_cfi.offlineBeamSpot.clone()



# Reject outliers <---- WARNING !!!! Applied only inside a TrackRefitter/TrackProducer, not by AlignmentTrackSelector
## import TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi 
##FittingSmootherCustomised = TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi.KFFittingSmootherWithOutliersRejectionAndRK.clone(


#####################################################################################################
### CTF tracks specialities 
#####################################################################################################

#0: filter on L1 trigger bit - if you really want to, remember to add to the path the entry "l1Filter"

import L1Trigger.Configuration.L1Config_cff
#import L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu_CRUZET200805_gr7_muon_cff ###WHAT IS THE RIGTH ONE ????
import L1Trigger.Skimmer.l1Filter_cfi
ALCAl1Filter =  L1Trigger.Skimmer.l1Filter_cfi.l1Filter.clone(
    #algorithms=('L1_DoubleMuTopBottom')
    )





#1: first refit to the tracks, needed for getting the Traj

from RecoTracker.TrackProducer.TrackRefitters_cff import *
#import TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
#FittingSmootherCustomised = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.FittingSmootherRK.clone(
FittingSmootherCustomised =TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'FittingSmootherCustomised',
    EstimateCut=18.0,
    MinNumberOfHits=6
    )


TrackRefitterCTF1 =RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    constraint = "",
    src='ALCARECOTkAlMinBias',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True,
    beamSpot='MyBeamSpot'
    )

# 2b: apply NEW hit filter. Does not work with CosmicTF tracks !

from RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff import *
AlignmentHitFilterCTF=RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff.TrackerTrackHitFilter.clone(
  ## src = 'ALCARECOTkAlCosmicsCTF0T',
    src = 'TrackRefitterCTF1',
    commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC"),
    minimumHits = 6,
    replaceWithInactiveHits = True,
    stripAllInvalidHits = False,
    rejectBadStoNHits = True,
    StoNcommands = cms.vstring("ALL 18.0"),
    useTrajectories= True,
    rejectLowAngleHits= True,
    TrackAngleCut= 0.17,
    usePixelQualityFlag= True,
    PxlCorrClusterChargeCut=10000.0
    )

# 3: produce track after NEW track hit filter

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff
ctfProducerCustomisedCTF = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
    src = 'AlignmentHitFilterCTF',
    beamSpot='MyBeamSpot',
  #  Fitter = 'FittingSmootherCustomised',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True
    )


# 4: apply track selections on the refitted tracks
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMinBiasSkimmed= Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
  ## src= 'TrackRefitterCTF1',
    src= 'ctfProducerCustomisedCTF',
    ptMin=1.5, # already in ALCARECO cfg
    ptMax=9999.0,
    pMin=3.0,
    pMax=9999.0,
    etaMin=-2.4,  # already in ALCARECO cfg
    etaMax=2.4,   # already in ALCARECO cfg
    nHitMin=8,
    nHitMin2D=2,
    chi2nMax=6.0
    ### others which aren't used
    #minHitsPerSubDet.inTIB = 0
    #minHitsPerSubDet.inBPIX = 1
    )


TrackRefitterCTF2 =RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    constraint = "",
    src='ALCARECOTkAlMinBiasSkimmed',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True,
    beamSpot='MyBeamSpot',
#    EstimateCut=15.0,
#    MinNumberOfHits=6
#    Fitter='FittingSmootherCustomised'
    ) 


# 5: Overlap tagger
import Alignment.TrackerAlignment.TkAlCaOverlapTagger_cff
OverlapAssoMapCTF=Alignment.TrackerAlignment.TkAlCaOverlapTagger_cff.OverlapTagger.clone(
  #  src='ALCARECOTkAlCosmicsCTFSkimmed'
    src='TrackRefitterCTF2',
    #Clustersrc='ALCARECOTkAlCosmicsCTF0T'
    Clustersrc='ALCARECOTkAlMinBiasSkimmed'#the track selector produces a new collection of Clusters!
    )


import Alignment.CommonAlignmentMonitor.AlignmentStats_cff
NewStatsCTF=Alignment.CommonAlignmentMonitor.AlignmentStats_cff.AlignmentStats.clone(
  #  src='OverlapAssoMap',
    src='TrackRefitterCTF2',
    OverlapAssoMap='OverlapAssoMapCTF',
    keepTrackStats = False,
    keepHitStats = True,
    TrkStatsFileName='TracksStatisticsCTF.root',
    HitStatsFileName='HitMapsCTF.root',
    TrkStatsPrescale= 1                            
    )



##________________________________Sequences____________________________________
seqALCARECOTkAlMinBiasSkimmed = cms.Sequence(MyBeamSpot+TrackRefitterCTF1+AlignmentHitFilterCTF+ctfProducerCustomisedCTF+ALCARECOTkAlMinBiasSkimmed+TrackRefitterCTF2+OverlapAssoMapCTF)



