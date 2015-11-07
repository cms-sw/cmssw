import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------
# HIGH LEVEL RECO

# Tracking
from RecoHI.HiTracking.HiTracking_cff import *    # two additional steps

# Egamma
from RecoHI.HiEgammaAlgos.HiEgamma_cff import *
from RecoHI.HiEgammaAlgos.HiElectronSequence_cff import *
ecalDrivenElectronSeeds.SeedConfiguration.SCEtCut = cms.double(15.0)
ecalDrivenGsfElectrons.minSCEtBarrel = cms.double(15.0)
ecalDrivenGsfElectrons.minSCEtEndcaps = cms.double(15.0)

# Jet Reconstruction
from RecoHI.HiJetAlgos.HiRecoJets_cff import *

# Muon Reco
from RecoHI.HiMuonAlgos.HiRecoMuon_cff import * 
# keep regit seperate for the moment
from RecoHI.HiMuonAlgos.HiRegionalRecoMuon_cff import *

from RecoHI.Configuration.Reconstruction_hiPF_cff import *

# Heavy Ion Event Characterization
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import *
from RecoHI.HiCentralityAlgos.CentralityBin_cfi import *
from RecoHI.HiCentralityAlgos.HiClusterCompatibility_cfi import *
from RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi import *

# HCAL noise producer
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
hcalnoise.trackCollName = 'hiGeneralTracks'

# Global + High-Level Reco Sequence
globalRecoPbPb = cms.Sequence(hiTracking_wSplitting
                              * hiParticleFlowLocalReco
                              * hiEcalClusters
                              * hiRecoJets
                              * muonRecoPbPb
                              * hiElectronSequence 
                              * hiEgammaSequence
                              * hiParticleFlowReco
                              * hiCentrality
                              * centralityBin
                              * hiClusterCompatibility
                              * hiEvtPlane
                              * hcalnoise
                              * muonRecoHighLevelPbPb
                              )

globalRecoPbPb_wConformalPixel = cms.Sequence(hiTracking_wConformalPixel
                                              * hiParticleFlowLocalReco
                                              * hiEcalClusters
                                              * hiRecoJets
                                              * muonRecoPbPb
                                              * hiElectronSequence
                                              * hiEgammaSequence
                                              * hiParticleFlowReco
                                              * hiCentrality
                                              * centralityBin
                                              * hiClusterCompatibility
                                              * hiEvtPlane
                                              * hcalnoise
                                              * muonRecoHighLevelPbPb
                                              )

#--------------------------------------------------------------------------
# Full sequence (LOCAL RECO + HIGH LEVEL RECO) 
# in Configuration.StandardSequences.ReconstructionHeavyIons_cff

# Modify zero-suppression sequence here
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
siStripZeroSuppression.storeCM = cms.bool(True)
