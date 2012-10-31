import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllMuons_cfi  import *
#from CommonTools.ParticleFlow.ParticleSelectors.pfMuonsPtGt5_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfMuonsFromVertex_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfSelectedMuons_cfi import *
from CommonTools.ParticleFlow.Isolation.pfMuonIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedMuons_cfi import *

pfMuons = pfIsolatedMuons.clone()
pfMuons.isolationCut = 999

pfMuonSequence = cms.Sequence(
    pfAllMuons +
    # muon selection
    #pfMuonsPtGt5 +
    pfMuonsFromVertex +
    pfSelectedMuons +
    # computing isolation variables:
    pfMuonIsolationSequence + 
    # selecting isolated electrons:
    pfIsolatedMuons+
    pfMuons 
    )




