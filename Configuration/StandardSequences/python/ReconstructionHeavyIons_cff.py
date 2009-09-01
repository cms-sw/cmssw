import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------
# LOCAL RECO

# Tracker
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *

# Ecal
from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *

# Hcal
from RecoLocalCalo.Configuration.hcalLocalReco_cff import *

# Muons
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
from RecoLuminosity.LumiProducer.lumiProducer_cfi import *

#--------------------------------------------------------------------------
# HIGH LEVEL RECO

from RecoHI.Configuration.Reconstruction_HI_cff import *

#--------------------------------------------------------------------------

caloReco = cms.Sequence(ecalLocalRecoSequence*hcalLocalRecoSequence)
muonReco = cms.Sequence(trackerlocalreco+muonlocalreco+lumiProducer)
localReco = cms.Sequence(offlineBeamSpot*muonReco*caloReco)

#--------------------------------------------------------------------------
# Main Sequence

reconstruct_PbPb = cms.Sequence(localReco*heavyIonTracking*muontracking_with_TeVRefinement*hiEcalClusters*runjets*hiEgammaSequence*hiCentrality*hiEvtPlane)
reconstruct_PbPb_CaloOnly = cms.Sequence(caloReco*hiEcalClusters*runjets*hiCentrality*hiEvtPlane)
reconstruct_PbPb_MuonOnly = cms.Sequence(offlineBeamSpot*muonReco*heavyIonTracking*muontracking_with_TeVRefinement)

reconstruction = cms.Sequence(reconstruct_PbPb)

#--------------------------------------------------------------------------
