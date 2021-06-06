import FWCore.ParameterSet.Config as cms

##################################################################
# AlCaReco for track based monitoring using single muon events
##################################################################
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOSiPixelCalSingleMuonTightHLTFilter = hltHighLevel.clone()
ALCARECOSiPixelCalSingleMuonTightHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECOSiPixelCalSingleMuonTightHLTFilter.throw = False ## dont throw on unknown path names
ALCARECOSiPixelCalSingleMuonTightHLTFilter.HLTPaths = ["HLT_*"]
#ALCARECOSiPixelCalSingleMuonTightHLTFilter.eventSetupPathsKey = 'SiPixelCalSingleMuonTight'  ## FIXME: to be changed once trigger bit is updated

##################################################################
# Filter on the DCS partitions
##################################################################
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOSiPixelCalSingleMuonTightDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX',
                               'DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

##################################################################
# Isolated muons Track selector
##################################################################
import Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi
ALCARECOSiPixelCalSingleMuonTightGoodMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlGoodIdMuonSelector.clone()
ALCARECOSiPixelCalSingleMuonTightRelCombIsoMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlRelCombIsoMuonSelector.clone(
    src = 'ALCARECOSiPixelCalSingleMuonTightGoodMuons'
)

##################################################################
# Basic Track selection
##################################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiPixelCalSingleMuonTight = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    filter = True, ##do not store empty events
    applyBasicCuts = True,
    ptMin = 2.0, ##GeV 
    etaMin = -3.5,
    etaMax = 3.5,
    nHitMin = 0
)

##################################################################
# Muon selection
##################################################################
ALCARECOSiPixelCalSingleMuonTight.GlobalSelector.muonSource = 'ALCARECOSiPixelCalSingleMuonTightRelCombIsoMuons'
# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
ALCARECOSiPixelCalSingleMuonTight.GlobalSelector.applyIsolationtest = False
ALCARECOSiPixelCalSingleMuonTight.GlobalSelector.minJetDeltaR = 0.1
ALCARECOSiPixelCalSingleMuonTight.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOSiPixelCalSingleMuonTight.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOSiPixelCalSingleMuonTight.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOSiPixelCalSingleMuonTight.TwoBodyDecaySelector.applyAcoplanarityFilter = False

##################################################################
# Track refitter
##################################################################
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
#from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *

ALCARECOSiPixelCalSingleMuonTightTracksRefit = TrackRefitter.clone(src = cms.InputTag("ALCARECOSiPixelCalSingleMuonTight"),
                                                                   NavigationSchool = cms.string("")
                                                                   )

##################################################################
# Producer or close-by-pixels
##################################################################
import Calibration.TkAlCaRecoProducers.NearbyPixelClustersProducer_cfi as NearbyPixelClusters
closebyPixelClusters = NearbyPixelClusters.NearbyPixelClustersProducer.clone(clusterCollection = 'siPixelClusters',
                                                                             trajectoryInput = 'ALCARECOSiPixelCalSingleMuonTightTracksRefit')

##################################################################
# Sequence: track refit + close-by-pixel producer
##################################################################
ALCARECOSiPixelCalSingleMuonTightOffTrackClusters = cms.Sequence(ALCARECOSiPixelCalSingleMuonTightTracksRefit +
                                                                 closebyPixelClusters)

##################################################################
# Producer of distances value map
##################################################################
import Calibration.TkAlCaRecoProducers.TrackDistanceValueMapProducer_cfi as TrackDistanceValueMap 
trackDistances = TrackDistanceValueMap.TrackDistanceValueMapProducer.clone(muonTracks = 'ALCARECOSiPixelCalSingleMuonTight')

##################################################################
# Final Tight sequence
##################################################################
seqALCARECOSiPixelCalSingleMuonTight = cms.Sequence(offlineBeamSpot+
                                                    ALCARECOSiPixelCalSingleMuonTightHLTFilter+
                                                    ALCARECOSiPixelCalSingleMuonTightDCSFilter+
                                                    ALCARECOSiPixelCalSingleMuonTightGoodMuons+
                                                    ALCARECOSiPixelCalSingleMuonTightRelCombIsoMuons+
                                                    ALCARECOSiPixelCalSingleMuonTight+
                                                    trackDistances +
                                                    ALCARECOSiPixelCalSingleMuonTightOffTrackClusters)
