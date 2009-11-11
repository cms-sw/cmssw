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
import L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu_CRUZET200805_gr7_muon_cff ###WHAT IS THE RIGTH ONE ????
import L1Trigger.Skimmer.l1Filter_cfi
ALCAl1Filter =  L1Trigger.Skimmer.l1Filter_cfi.l1Filter.clone(
    #algorithms=('L1_DoubleMuTopBottom')
    )





#1: first refit to the tracks, needed for getting the Traj

from RecoTracker.TrackProducer.TrackRefitters_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
FittingSmootherCustomised = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.FittingSmootherRKP5.clone(
    ComponentName = 'FittingSmootherCustomised',
    EstimateCut=15.0,
    MinNumberOfHits=6
    )


TrackRefitterCTF1 =RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
    constraint = "",
    src='ALCARECOTkAlCosmicsCTF0T',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True,
    beamSpot='MyBeamSpot',
    NavigationSchool=''
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
    TrackAngleCut= 0.35,
    usePixelQualityFlag= True,
    PxlCorrClusterChargeCut=10000.0
    )

# 3: produce track after NEW track hit filter

import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
ctfProducerCustomisedCTF = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = 'AlignmentHitFilterCTF',
    beamSpot='MyBeamSpot',
  #  Fitter = 'FittingSmootherCustomised',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True
    )


# 4: apply track selections on the refitted tracks
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlCosmicsCTF4TSkimmed= Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
  ## src= 'TrackRefitterCTF1',
    src= 'ctfProducerCustomisedCTF',
    ptMin=0.0,
    ptMax=9999.0,
    pMin=4.0,
    pMax=9999.0,
    etaMin=-99.0,
    etaMax=99.0,
    nHitMin=8,
    nHitMin2D=2,
    chi2nMax=6.0
    ### others which aren't used
    #minHitsPerSubDet.inTIB = 0
    #minHitsPerSubDet.inBPIX = 1
    )


TrackRefitterCTF2 =RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
    constraint = "",
    src='ALCARECOTkAlCosmicsCTF4TSkimmed',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True,
    beamSpot='MyBeamSpot',
    NavigationSchool=''
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
    Clustersrc='ALCARECOTkAlCosmicsCTF4TSkimmed'#the track selector produces a new collection of Clusters!
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

##seqALCARECOTkAlCosmicsCTFSkimmed = cms.Sequence(MyBeamSpot+TrackHitFilterCTF+TrackRefitterCTF1+ALCARECOTkAlCosmicsCTF0TSkimmed+TrackRefitterCTF2+OverlapAssoMapCTF+NewStatsCTF)
#seqALCARECOTkAlCosmicsCTFSkimmed = cms.Sequence(MyBeamSpot+TrackRefitterCTF1+AlignmentHitFilterCTF+ctfProducerCustomisedCTF+ALCARECOTkAlCosmicsCTF4TSkimmed+TrackRefitterCTF2+OverlapAssoMapCTF+NewStatsCTF)
seqALCARECOTkAlCosmicsCTFSkimmed = cms.Sequence(MyBeamSpot+TrackRefitterCTF1+AlignmentHitFilterCTF+ctfProducerCustomisedCTF+ALCARECOTkAlCosmicsCTF4TSkimmed+TrackRefitterCTF2+OverlapAssoMapCTF)

