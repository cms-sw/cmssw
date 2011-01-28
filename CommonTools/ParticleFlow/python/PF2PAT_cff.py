import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from CommonTools.ParticleFlow.pfNoPileUp_cff  import *
from CommonTools.ParticleFlow.pfElectrons_cff import *
from CommonTools.ParticleFlow.pfMuons_cff import *
from CommonTools.ParticleFlow.pfJets_cff import *
from CommonTools.ParticleFlow.pfTaus_cff import *

# sequential top projection cleaning
from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *
from CommonTools.ParticleFlow.TopProjectors.pfNoMuon_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoElectron_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cfi import *

# generator tools
from CommonTools.ParticleFlow.genForPF2PAT_cff import *


PF2PAT = cms.Sequence(
    pfMET +
    pfNoPileUpSequence + 
    # pfSortByTypeSequence +
    pfAllNeutralHadrons+
    pfAllChargedHadrons+
    pfAllPhotons+
    # pfAllMuons + in 'pfMuonSequence' 
    pfMuonSequence + 
    pfNoMuon +
    # pfAllElectrons + in 'pfElectronSequence' 
    pfElectronSequence +
    pfNoElectron + 
# when uncommenting, change the source of the jet clustering
    pfJetSequence +
    pfNoJet + 
    pfTauSequence +
    pfNoTau 
# now that we have real data, we leave it to the user
# or maybe to PAT? to run the gen sequence. 
    )
