import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from CommonTools.ParticleFlow.pfNoPileUp_cff  import *
from CommonTools.ParticleFlow.pfPhotons_cff import *
from CommonTools.ParticleFlow.pfElectrons_cff import *
from CommonTools.ParticleFlow.pfMuons_cff import *
from CommonTools.ParticleFlow.pfJets_cff import *
from CommonTools.ParticleFlow.pfTaus_cff import *

# sequential top projection cleaning
from CommonTools.ParticleFlow.TopProjectors.pfNoMuon_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoElectron_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cff import *
from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cff import *

# getting the ptrs
from RecoParticleFlow.PFProducer.pfLinker_cff import particleFlowPtrs

# generator tools
from CommonTools.ParticleFlow.genForPF2PAT_cff import *

# plugging PF2PAT on the collection of PFCandidates from RECO:
#particleFlowPtrs.src = 'particleFlow'

pfPileUp.PFCandidates = 'particleFlowPtrs'
pfNoPileUp.bottomCollection = 'particleFlowPtrs'
pfPileUpIso.PFCandidates = 'particleFlowPtrs' 
pfNoPileUpIso.bottomCollection='particleFlowPtrs'
pfPileUpJME.PFCandidates = 'particleFlowPtrs' 
pfNoPileUpJME.bottomCollection='particleFlowPtrs'

PFBRECO = cms.Sequence(
    particleFlowPtrs +
    pfNoPileUpSequence +
    pfNoPileUpJMESequence +
    pfParticleSelectionSequence + 
    pfPhotonSequence +
    pfMuonSequence + 
    pfNoMuon +
    pfNoMuonJME +
    pfElectronSequence +
    pfNoElectron +
    pfNoElectronJME +
    pfNoElectronJMEClones+
    pfJetSequence +
    pfNoJet + 
    pfTauSequence +
    pfNoTau +
    pfMET
    )
