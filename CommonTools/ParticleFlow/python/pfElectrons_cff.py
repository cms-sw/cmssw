import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi  import *
#from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsPtGt5_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsFromVertex_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfSelectedElectrons_cfi import *
from CommonTools.ParticleFlow.Isolation.pfElectronIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedElectrons_cfi import *


pfElectronsClones = pfIsolatedElectronsClones.clone()
pfElectronsClones.isolationCut = 999

pfElectrons = pfIsolatedElectrons.clone(src=cms.InputTag("pfElectronsClones"))

pfElectronSequence = cms.Sequence(
    pfAllElectrons +
    pfAllElectronsClones +
    # electron selection:
    #pfElectronsPtGt5 +
    pfElectronsFromVertex +
    pfSelectedElectrons +
    # computing isolation variables:
    pfElectronIsolationSequence +
    # selecting isolated electrons:
    pfIsolatedElectronsClones+
    pfElectronsClones+
    pfIsolatedElectrons +
    pfElectrons 
    )




