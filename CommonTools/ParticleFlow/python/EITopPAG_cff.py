import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from CommonTools.ParticleFlow.pfJets_cff import *
from CommonTools.ParticleFlow.pfTaus_cff import *
from CommonTools.ParticleFlow.pfMuons_cff import *

from RecoParticleFlow.PFProducer.electronPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedElectrons_cfi import *

pfBasedElectronIsoSequence = cms.Sequence(
    electronPFIsolationDepositsSequence +
    electronPFIsolationValuesSequence
    )

# sequential top projection cleaning
from CommonTools.ParticleFlow.TopProjectors.pfNoMuon_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoElectron_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cfi import *


EITopPAG = cms.Sequence(
    pfMuonSequence +
    pfNoMuon +
    pfNoMuonJME +
    pfBasedElectronIsoSequence +
    pfIsolatedElectrons + 
    pfNoElectron +
    pfNoElectronJME +
    pfNoElectronJMEClones+
    pfJetSequence +
    pfNoJet + 
    pfTauSequence +
    pfNoTau +
    pfMET
    )

