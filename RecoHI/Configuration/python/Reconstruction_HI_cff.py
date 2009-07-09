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

#--------------------------------------------------------------------------
# HIGH LEVEL RECO

# Tracking
from RecoHI.HiTracking.HighPtTracking_PbPb_cff import *

# Egamma
from RecoHI.HiEgammaAlgos.HiEgamma_cff import *

# Jet Reconstruction
from RecoHI.HiJetAlgos.IterativeConePu5Jets_PbPb_cff import *

# Muon Reco
from RecoMuon.Configuration.RecoMuon_cff import *

# Heavy Ion Event Characterization
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import *
from RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi import *

#--------------------------------------------------------------------------

caloReco = cms.Sequence(ecalLocalRecoSequence*hcalLocalRecoSequence)
localReco = cms.Sequence(offlineBeamSpot*trackerlocalreco*caloReco)

#--------------------------------------------------------------------------
# Main Sequence
reconstruct_PbPb_CaloOnly = cms.Sequence(caloReco*hiEcalClusters*runjets*hiCentrality*hiEvtPlane)
reconstruct_PbPb = cms.Sequence(localReco*heavyIonTracking*hiEcalClusters*runjets*hiCentrality*hiEvtPlane)

#--------------------------------------------------------------------------

