import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------
# HIGH LEVEL RECO

# Tracking
from RecoHI.HiTracking.HiTracking_cff import *    # two additional steps

# Egamma
from RecoHI.HiEgammaAlgos.HiEgamma_cff import *
from RecoHI.HiEgammaAlgos.HiElectronSequence_cff import *

# Jet Reconstruction
from RecoHI.HiJetAlgos.HiRecoJets_cff import *

# Muon Reco
from RecoHI.HiMuonAlgos.HiRecoMuon_cff import * 
# keep regit seperate for the moment
from RecoHI.HiMuonAlgos.HiRegionalRecoMuon_cff import *

from RecoHI.Configuration.Reconstruction_hiPF_cff import *

# Heavy Ion Event Characterization
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import *
from RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi import *

# HCAL noise producer
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
hcalnoise.trackCollName = 'hiGeneralTracks'

# Global + High-Level Reco Sequence
globalRecoPbPb = cms.Sequence(hiTracking
                              * hiEcalClusters
                              * hiRecoJets
                              * muonRecoPbPb
                              * hiElectronSequence
                              * hiEgammaSequence
                              * HiParticleFlowReco
                              * hiCentrality
                              * hiEvtPlane
                              * hcalnoise
                              )

globalRecoPbPb_wConformalPixel = cms.Sequence(hiTracking_wConformalPixel
                              * hiEcalClusters
                              * hiRecoJets
                              * muonRecoPbPb
                              * hiElectronSequence
                              * HiParticleFlowLocalReco
                              * hiEgammaSequence
                              * HiParticleFlowReco
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

