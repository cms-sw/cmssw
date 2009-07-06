import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.blockHLT_8E29_cff import *

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
from RecoMuon.TrackerSeedGenerator.TrackerSeedCleaner_cff import *
# include  "RecoMuon/TrackerSeedGenerator/data/TSGs.cff"


def l3seeds(old):
    if (old):
        return cms.EDFilter("FastTSGFromL2Muon",
                            # ServiceParameters
                            MuonServiceProxy,
                            # The collection of Sim Tracks
                            SimTrackCollectionLabel = cms.InputTag("famosSimHits"),
                            # The STA muons for which seeds are looked for in the tracker
                            MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
                            #using TrackerSeedCleanerCommon
                            MuonTrackingRegionBuilder = cms.PSet(
                             block_hltL3TrajectorySeed
                            ),
                            # Keep tracks with pT > 1 GeV 
                            PtCut = cms.double(1.0),
                            # The Tracks from which seeds are looked for
                            SeedCollectionLabels = cms.VInputTag(cms.InputTag("pixelTripletSeeds","PixelTriplet"), cms.InputTag("globalPixelSeeds","GlobalPixel"))
                            )
    else:
        return cms.EDProducer("TSGFromL2Muon",
    tkSeedGenerator = cms.string('TSGForRoadSearchOI'),
    TSGFromCombinedHits = cms.PSet(    ),
    ServiceParameters = cms.PSet(
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True),
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorOpposite', 
            'SteppingHelixPropagatorAlong')
    ),
    TSGFromPropagation = cms.PSet(    ),
    TSGFromPixelTriplets = cms.PSet(    ),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    TSGForRoadSearchOI = cms.PSet(
#    TkSeedGenerator = cms.PSet(
        propagatorCompatibleName = cms.string('SteppingHelixPropagatorOpposite'),
        option = cms.uint32(3),
        maxChi2 = cms.double(40.0),
        errorMatrixPset = cms.PSet(
            atIP = cms.bool(True),
            action = cms.string('use'),
            errorMatrixValuesPSet = cms.PSet(
                pf3_V12 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                pf3_V13 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                pf3_V11 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(3, 3, 3, 5, 4, 
                        5, 10, 7, 10, 10, 
                        10, 10)
                ),
                pf3_V45 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                pf3_V14 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                yAxis = cms.vdouble(0, 1.0, 1.4, 10),
                pf3_V15 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                pf3_V35 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                zAxis = cms.vdouble(-3.14159, 3.14159),
                pf3_V44 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(3, 3, 3, 5, 4, 
                        5, 10, 7, 10, 10, 
                        10, 10)
                ),
                xAxis = cms.vdouble(0, 13, 30, 70, 1000),
                pf3_V23 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                pf3_V22 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(3, 3, 3, 5, 4, 
                        5, 10, 7, 10, 10, 
                        10, 10)
                ),
                pf3_V55 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(3, 3, 3, 5, 4, 
                        5, 10, 7, 10, 10, 
                        10, 10)
                ),
                pf3_V34 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                pf3_V33 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(3, 3, 3, 5, 4, 
                        5, 10, 7, 10, 10, 
                        10, 10)
                ),
                pf3_V25 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                ),
                pf3_V24 = cms.PSet(
                    action = cms.string('scale'),
                    values = cms.vdouble(1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 
                        1, 1)
                )
            )
        ),
        propagatorName = cms.string('SteppingHelixPropagatorAlong'),
        manySeeds = cms.bool(False),
        copyMuonRecHit = cms.bool(False),
        ComponentName = cms.string('TSGForRoadSearch')
    ),
    MuonTrackingRegionBuilder = cms.PSet(    ),
    TSGFromMixedPairs = cms.PSet(    ),
    PCut = cms.double(2.5),
    TrackerSeedCleaner = cms.PSet(    ),
    PtCut = cms.double(1.0),
    TSGForRoadSearchIOpxl = cms.PSet(    ),
    TSGFromPixelPairs = cms.PSet(    )
                              
)
                                     
hltL3TrajectorySeed = l3seeds(False)

