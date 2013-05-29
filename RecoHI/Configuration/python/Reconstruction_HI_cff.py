import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------
# HIGH LEVEL RECO

# Tracking
#from RecoHI.HiTracking.HighPtTracking_PbPb_cff import *  # above 1.5 GeV
from RecoHI.HiTracking.LowPtTracking_PbPb_cff import *    # above 0.9 GeV

# Egamma
from RecoHI.HiEgammaAlgos.HiEgamma_cff import *

# Jet Reconstruction
from RecoHI.HiJetAlgos.HiRecoJets_cff import *

# Muon Reco
from RecoHI.HiMuonAlgos.HiRecoMuon_cff import * 

# Regit Muon Reco
from RecoHI.HiMuonAlgos.HiReRecoMuon_cff import *

# Heavy Ion Event Characterization
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import *
from RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi import *

# HCAL noise producer
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
hcalnoise.trackCollName = 'hiSelectedTracks'

# Global + High-Level Reco Sequence
globalRecoPbPb = cms.Sequence(heavyIonTracking
                              * hiEcalClusters
                              * hiRecoJets
                              * muonRecoPbPb
                              * hiEgammaSequence
                              * hiCentrality
                              * hiEvtPlane
                              * hcalnoise
                              )


#--------------------------------------------------------------------------
# Full sequence (LOCAL RECO + HIGH LEVEL RECO) 
# in Configuration.StandardSequences.ReconstructionHeavyIons_cff

# Modify zero-suppression sequence here
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
siStripZeroSuppression.storeCM = cms.bool(True)

