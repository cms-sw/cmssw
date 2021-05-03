import FWCore.ParameterSet.Config as cms

from ..modules.L1TkMuons_cfi import *
from ..modules.L1TkPrimaryVertex_cfi import *
from ..modules.l1tSlwPFPuppiJetsCorrected_cfi import *
from ..modules.l1tSlwPFPuppiJets_cfi import *
from ..modules.simEmtfDigis_cfi import *
from ..modules.simKBmtfDigis_cfi import *
from ..modules.simKBmtfStubs_cfi import *
from ..modules.simOmtfDigis_cfi import *
from ..modules.simTwinMuxDigis_cfi import *
from ..modules.L1EGammaClusterEmuProducer_cfi import *
from ..modules.L1TkPhotonsCrystal_cfi import *
from ..modules.L1TkElectronsEllipticMatchCrystal_cfi import *
from ..modules.l1EGammaEEProducer_cfi import *
from ..modules.L1TkElectronsEllipticMatchHGC_cfi import *
from ..modules.L1TkPhotonsHGC_cfi import *

l1tReconstructionTask = cms.Task(L1TkMuons, L1TkPrimaryVertex, l1tSlwPFPuppiJets, l1tSlwPFPuppiJetsCorrected, simEmtfDigis, simKBmtfDigis, simKBmtfStubs, simOmtfDigis, simTwinMuxDigis, L1EGammaClusterEmuProducer, L1TkElectronsEllipticMatchCrystal, L1TkPhotonsCrystal, l1EGammaEEProducer, L1TkElectronsEllipticMatchHGC, L1TkPhotonsHGC)
