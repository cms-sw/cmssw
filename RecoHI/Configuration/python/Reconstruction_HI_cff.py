import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------
# HIGH LEVEL RECO

# Tracking
from RecoHI.HiTracking.HiTracking_cff import *    # two additional steps

# Egamma
from RecoHI.HiEgammaAlgos.HiEgamma_cff import *
from RecoHI.HiEgammaAlgos.HiElectronSequence_cff import *
ecalDrivenElectronSeeds.SeedConfiguration.SCEtCut = cms.double(15.0)
ecalDrivenGsfElectrons.preselection.minSCEtBarrel = cms.double(15.0)
ecalDrivenGsfElectrons.preselection.minSCEtEndcaps = cms.double(15.0)

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

from RecoLocalCalo.Configuration.hcalGlobalReco_cff import *

#post PF egamma stuff
from RecoHI.HiEgammaAlgos.HiEgammaPostPF_cff import *

from RecoHI.HiJetAlgos.HiRecoPFJets_cff import *

#reduced rechits
from RecoEcal.EgammaClusterProducers.reducedRecHitsSequence_cff import *
from RecoEcal.EgammaCoreTools.EcalNextToDeadChannelESProducer_cff import *
from RecoLocalCalo.HcalRecProducers.HcalHitSelection_cfi import *
reducedHcalRecHitsTask = cms.Task( reducedHcalRecHits )
reducedHcalRecHitsSequence = cms.Sequence( reducedHcalRecHitsTask )
reducedRecHitsTask = cms.Task ( reducedEcalRecHitsTask , reducedHcalRecHitsTask )
reducedRecHits = cms.Sequence ( reducedRecHitsTask )
interestingTrackEcalDetIds.TrackCollection = "hiGeneralTracks"


# Global + High-Level Reco Sequence
globalRecoPbPbTask = cms.Task(hiTracking_wSplittingTask
                              , hcalGlobalRecoTask
                              , hiParticleFlowLocalRecoTask
                              , hiEcalClustersTask
                              , hiRecoJetsTask
                              , muonRecoPbPbTask
                              , hiElectronTask 
                              , hiEgammaTask
                              , hiParticleFlowRecoTask
                              , egammaHighLevelRecoPostPFTask
                              , hiCentrality
                              #, centralityBin  # temporarily removed
                              , hiClusterCompatibility
                              , hiEvtPlane
                              , hcalnoise
                              , muonRecoHighLevelPbPbTask
                              , particleFlowLinksTask
                              , hiRecoPFJetsTask
                              , reducedRecHitsTask
                              )
globalRecoPbPb = cms.Sequence(globalRecoPbPbTask)

globalRecoPbPb_wPhase1Task = globalRecoPbPbTask.copy()
globalRecoPbPb_wPhase1Task.replace(hiTracking_wSplittingTask, hiTracking_wSplitting_Phase1Task)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toReplaceWith(globalRecoPbPbTask, globalRecoPbPb_wPhase1Task)


globalRecoPbPb_wConformalPixelTask = cms.Task(hiTracking_wConformalPixelTask
                                              , hiParticleFlowLocalRecoTask
                                              , hiEcalClustersTask
                                              , hiRecoJetsTask
                                              , muonRecoPbPbTask
                                              , hiElectronTask
                                              , hiEgammaTask
                                              , hiParticleFlowRecoTask
                                              , egammaHighLevelRecoPostPFTask
                                              , hiCentrality
                                              , hiClusterCompatibility
                                              , hiEvtPlane
                                              , hcalnoise
                                              , muonRecoHighLevelPbPbTask
                                              , particleFlowLinksTask
                                              , hiRecoPFJetsTask
                                              , reducedRecHitsTask
                                              )
globalRecoPbPb_wConformalPixel = cms.Sequence(globalRecoPbPb_wConformalPixelTask)

#--------------------------------------------------------------------------
# Full sequence (LOCAL RECO + HIGH LEVEL RECO) 
# in Configuration.StandardSequences.ReconstructionHeavyIons_cff

# Modify zero-suppression sequence here
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
siStripZeroSuppression.storeCM = cms.bool(True)
