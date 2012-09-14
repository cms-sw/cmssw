import FWCore.ParameterSet.Config as cms

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
from RecoMuon.TrackerSeedGenerator.TrackerSeedCleaner_cff import *
from RecoMuon.TrackerSeedGenerator.TSGs_cff import *
# include  "RecoMuon/TrackerSeedGenerator/data/TSGs.cff"

def makeOIStateSet():
    return  cms.PSet(
        propagatorCompatibleName = cms.string('SteppingHelixPropagatorOpposite'),
        option = cms.uint32(3),
        maxChi2 = cms.double(40.0),
      errorMatrixPset = cms.PSet( 
        atIP = cms.bool( True ),
        action = cms.string( "use" ),
        errorMatrixValuesPSet = cms.PSet( 
          pf3_V12 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V13 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V11 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V14 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V15 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V34 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
          pf3_V33 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V45 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V44 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
          pf3_V23 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V22 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V55 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          zAxis = cms.vdouble( -3.14159, 3.14159 ),
          pf3_V35 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V25 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V24 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          )
        )
      ),
      propagatorName = cms.string('SteppingHelixPropagatorAlong'),
      manySeeds = cms.bool(False),
      copyMuonRecHit = cms.bool(False),
      ComponentName = cms.string('TSGForRoadSearch')
    )

def makeOIHitSet():
    return  cms.PSet(
      errorMatrixPset = cms.PSet( 
        atIP = cms.bool( True ),
        action = cms.string( "use" ),
        errorMatrixValuesPSet = cms.PSet( 
          pf3_V12 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V13 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V11 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V14 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V15 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V34 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
          pf3_V33 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V45 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V44 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
          pf3_V23 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V22 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V55 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          zAxis = cms.vdouble( -3.14159, 3.14159 ),
          pf3_V35 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V25 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V24 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          )
        )
      ),
      ComponentName = cms.string( "FastTSGFromPropagation" ),
      beamSpot = cms.InputTag('offlineBeamSpot'),
      Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
      MaxChi2 = cms.double( 15.0 ),
      ResetMethod = cms.string("matrix"),
      ErrorRescaling = cms.double(3.0),
      SigmaZ = cms.double(25.0),       
      SimTrackCollectionLabel = cms.InputTag("famosSimHits"),
      HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
      UseVertexState = cms.bool( True ),
      UpdateState = cms.bool( True ), #fixme
      SelectState = cms.bool( False )
      )        

def OIStatePropagators(hltL3TrajectorySeed,pset):
    if (not hasattr(hltL3TrajectorySeed.ServiceParameters,"Propagators")):
        hltL3TrajectorySeed.ServiceParameters.Propagators = cms.untracked.vstring()
    hltL3TrajectorySeed.ServiceParameters.Propagators.append(pset.propagatorCompatibleName.value())
    hltL3TrajectorySeed.ServiceParameters.Propagators.append(pset.propagatorName.value())
 
def OIHitPropagators(hltL3TrajectorySeed,pset):
    if (not hasattr(hltL3TrajectorySeed.ServiceParameters,"Propagators")):
        hltL3TrajectorySeed.ServiceParameters.Propagators = cms.untracked.vstring()
    hltL3TrajectorySeed.ServiceParameters.Propagators.append('PropagatorWithMaterial')
    hltL3TrajectorySeed.ServiceParameters.Propagators.append(pset.Propagator.value())

def makeIOHitSet():
    return  cms.PSet(
        ComponentName = cms.string( "FastTSGFromIOHit" ),       
        PtCut = cms.double(1.0),
        # The Tracks from which seeds are looked for
        SeedCollectionLabels = cms.VInputTag(
        cms.InputTag("pixelTripletSeeds","PixelTriplet"),
        cms.InputTag("globalPixelSeeds","GlobalPixel")),
        SimTrackCollectionLabel = cms.InputTag("famosSimHits"),
        #Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
        )

def makeOIHitCascadeSet():
    return cms.PSet (
        ComponentName = cms.string('DualByL2TSG'),
        PSetNames = cms.vstring('skipTSG','iterativeTSG'),
        skipTSG = cms.PSet(    ),
        iterativeTSG = makeOIHitSet(),
        #iterativeTSG = TSGsBlock.TSGFromPropagation,        
        L3TkCollectionA = cms.InputTag('hltL3MuonsOIState'),
        )

def makeIOHitCascadeSet():
    return cms.PSet (
        ComponentName = cms.string('DualByL2TSG'),
        PSetNames = cms.vstring('skipTSG','iterativeTSG'),
        skipTSG = cms.PSet(    ),
        iterativeTSG = makeIOHitSet(),
        L3TkCollectionA = cms.InputTag('hltL3MuonsOICombined'),
        )

def l3seeds(tsg = "old"):
    if (tsg == "old"):
        return cms.EDProducer("FastTSGFromL2Muon",
                            # ServiceParameters
                            MuonServiceProxy,
                            # The collection of Sim Tracks
                            SimTrackCollectionLabel = cms.InputTag("famosSimHits"),
                            # The STA muons for which seeds are looked for in the tracker
                            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
                            #using TrackerSeedCleanerCommon
                            # Keep tracks with pT > 1 GeV 
                            PtCut = cms.double(1.0),
                            # The Tracks from which seeds are looked for
                            SeedCollectionLabels = cms.VInputTag(cms.InputTag("pixelTripletSeeds","PixelTriplet"), cms.InputTag("globalPixelSeeds","GlobalPixel"))
                            )
    elif( tsg == "OIState" ):
        return cms.EDProducer(
            "TSGFromL2Muon",
            MuonServiceProxy,
            MuonTrackingRegionBuilder = cms.PSet(),
            TrackerSeedCleaner = cms.PSet(),
            TkSeedGenerator = TSGsBlock.TSGForRoadSearchOI,
            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
            PtCut = cms.double(1.0),
            PCut = cms.double(2.5),
            #
#            tkSeedGenerator = cms.string('TSGForRoadSearchOI'),
#            TSGFromCombinedHits = cms.PSet( ),
#            ServiceParameters = cms.PSet(
#                RPCLayers = cms.bool(True),
#                UseMuonNavigation = cms.untracked.bool(True),
#                Propagators = cms.untracked.vstring('SteppingHelixPropagatorOpposite', 
#                    'SteppingHelixPropagatorAlong')
#                ),
#            TSGFromPropagation = cms.PSet(    ),
#            TSGFromPixelTriplets = cms.PSet(    ),
#            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
#            TSGForRoadSearchOI = makeOIStateSet(),
#            MuonTrackingRegionBuilder = cms.PSet(    ),
#            TSGFromMixedPairs = cms.PSet(    ),
#            PCut = cms.double(2.5),
#            TrackerSeedCleaner = cms.PSet(    ),
#            PtCut = cms.double(1.0),
#            TSGForRoadSearchIOpxl = cms.PSet(    ),
#            TSGFromPixelPairs = cms.PSet(    )
            )
    elif( tsg == "OIHit" ):
        return cms.EDProducer("TSGFromL2Muon",
            tkSeedGenerator = cms.string('FastTSGFromPropagation'),
            beamSpot = cms.InputTag('offlineBeamSpot'),
            TSGFromCombinedHits = cms.PSet(    ),
            ServiceParameters = cms.PSet(
               RPCLayers = cms.bool(True),
               UseMuonNavigation = cms.untracked.bool(True),
               Propagators = cms.untracked.vstring(
                  'SteppingHelixPropagatorOpposite', 
                  'SteppingHelixPropagatorAlong', 'PropagatorWithMaterial', 'hltESPSmartPropagatorAnyOpposite')
               ),
            TSGFromPixelTriplets = cms.PSet(    ),
            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
            TSGForRoadSearchOI = cms.PSet(),
            FastTSGFromPropagation = makeOIHitSet(),
            MuonTrackingRegionBuilder = cms.PSet(    ),
            TSGFromMixedPairs = cms.PSet(    ),
            PCut = cms.double(2.5),
            TrackerSeedCleaner = cms.PSet(    ),
            PtCut = cms.double(1.0),
            TSGForRoadSearchIOpxl = cms.PSet(    ),
            TSGFromPixelPairs = cms.PSet(    )
            )
    elif( tsg == "IOHit" ):
        return cms.EDProducer("TSGFromL2Muon",
            PCut = cms.double(2.5),
            PtCut = cms.double(1.0),
            tkSeedGenerator = cms.string('TSGFromCombinedHits'),
            ServiceParameters = cms.PSet(
               RPCLayers = cms.bool(True),
               UseMuonNavigation = cms.untracked.bool(True),
               Propagators = cms.untracked.vstring(
                  'SteppingHelixPropagatorOpposite', 
                  'SteppingHelixPropagatorAlong', 'PropagatorWithMaterial',
                  'hltESPSmartPropagatorAnyOpposite')
               ),
            TrackerSeedCleaner = cms.PSet(    ),
            MuonTrackingRegionBuilder = cms.PSet(
                EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
                Eta_fixed = cms.double( 0.2 ),
                beamSpot = cms.InputTag( "offlineBeamSpot" ),
                MeasurementTrackerName = cms.string( "" ),
                OnDemand = cms.double( -1.0 ),
                Rescale_Dz = cms.double( 3.0 ),
                Eta_min = cms.double( 0.1 ),
                Rescale_phi = cms.double( 3.0 ),
                PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
                DeltaZ_Region = cms.double( 15.9 ),
                Phi_min = cms.double( 0.1 ),
                PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
                vertexCollection = cms.InputTag( "pixelVertices" ),
                Phi_fixed = cms.double( 0.2 ),
                DeltaR = cms.double( 0.2 ),
                EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
                UseFixedRegion = cms.bool( False ),
                Rescale_eta = cms.double( 3.0 ),
                UseVertex = cms.bool( False ),
                EscapePt = cms.double( 1.5 )
            ),
            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
            TSGFromCombinedHits = makeIOHitSet(),
            )
    elif( tsg == "OIHitCascade"):
        return cms.EDProducer(
            "TSGFromL2Muon",
            MuonServiceProxy,
            MuonTrackingRegionBuilder = cms.PSet(),
            TrackerSeedCleaner = cms.PSet(),
            #TkSeedGenerator = TSGsBlock.TSGFromPropagation,
            TkSeedGenerator = makeOIHitCascadeSet(),
            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
            PtCut = cms.double(1.0),
            PCut = cms.double(2.5),
            ####
            #tkSeedGenerator = cms.string('TSGFromPropagation'),
            #TSGFromCombinedHits = cms.PSet(    ),
            #ServiceParameters = cms.PSet(
            #   RPCLayers = cms.bool(True),
            #   UseMuonNavigation = cms.untracked.bool(True),
            #   Propagators = cms.untracked.vstring(
            #      'SteppingHelixPropagatorOpposite', 
            #      'SteppingHelixPropagatorAlong', 'PropagatorWithMaterial',
            #      'hltESPSmartPropagatorAnyOpposite')
            #   ),
            #TSGFromPixelTriplets = cms.PSet(    ),
            #MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
            #TSGForRoadSearchOI = cms.PSet(),
            #TSGFromPropagation = makeOIHitCascadeSet(),
            #MuonTrackingRegionBuilder = cms.PSet(    ),
            #TSGFromMixedPairs = cms.PSet(    ),
            #PCut = cms.double(2.5),
            #TrackerSeedCleaner = cms.PSet(    ),
            #PtCut = cms.double(1.0),
            #TSGForRoadSearchIOpxl = cms.PSet(    ),
            #TSGFromPixelPairs = cms.PSet(    )
            )
    elif( tsg == "IOHitCascade"):
        return cms.EDProducer(
            "TSGFromL2Muon",
            MuonServiceProxy,
            #MuonTrackingRegionBuilder = cms.PSet(),
            TrackerSeedCleaner = cms.PSet(),
            #TkSeedGenerator = TSGsBlock.TSGFromCombinedHits,
            TkSeedGenerator = makeIOHitCascadeSet(),
            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
            PtCut = cms.double(1.0),
            PCut = cms.double(2.5),
            #####
            #PCut = cms.double(2.5),
            #PtCut = cms.double(1.0),
            #tkSeedGenerator = cms.string('TSGFromCombinedHits'),
            #ServiceParameters = cms.PSet(
            #   RPCLayers = cms.bool(True),
            #   UseMuonNavigation = cms.untracked.bool(True),
            #   Propagators = cms.untracked.vstring(
            #      'SteppingHelixPropagatorOpposite', 
            #      'SteppingHelixPropagatorAlong', 'PropagatorWithMaterial',
            #      'hltESPSmartPropagatorAnyOpposite')
            #   ),
            #TrackerSeedCleaner = cms.PSet(    ),
            MuonTrackingRegionBuilder = cms.PSet(
                EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
                Eta_fixed = cms.double( 0.2 ),
                beamSpot = cms.InputTag( "offlineBeamSpot" ),
                MeasurementTrackerName = cms.string( "" ),
                OnDemand = cms.double( -1.0 ),
                Rescale_Dz = cms.double( 3.0 ),
                Eta_min = cms.double( 0.1 ),
                Rescale_phi = cms.double( 3.0 ),
                PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
                DeltaZ_Region = cms.double( 15.9 ),
                Phi_min = cms.double( 0.1 ),
                PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
                vertexCollection = cms.InputTag( "pixelVertices" ),
                Phi_fixed = cms.double( 0.2 ),
                DeltaR = cms.double( 0.2 ),
                EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
                UseFixedRegion = cms.bool( False ),
                Rescale_eta = cms.double( 3.0 ),
                UseVertex = cms.bool( False ),
                EscapePt = cms.double( 1.5 )
            ),
            #MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
            #TSGFromCombinedHits = makeIOHitCascadeSet(),
            )
    
hltL3TrajectorySeed = l3seeds("OIState")
