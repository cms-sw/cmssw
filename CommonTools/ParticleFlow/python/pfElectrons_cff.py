import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi  import *
#from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsPtGt5_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsFromVertex_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfSelectedElectrons_cfi import *
from CommonTools.ParticleFlow.Isolation.pfElectronIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedElectrons_cfi import *

pfElectrons = pfIsolatedElectrons.clone()
pfElectrons.isolationCut = 999

pfElectronSequence = cms.Sequence(
    pfAllElectrons +
    # electron selection:
    #pfElectronsPtGt5 +
    pfElectronsFromVertex +
    pfSelectedElectrons +
    # computing isolation variables:
    pfElectronIsolationSequence +
    # selecting isolated electrons:
    pfIsolatedElectrons +
    pfElectrons 
    )




