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

l1tReconstructionTask = cms.Task(L1TkMuons, L1TkPrimaryVertex, l1tSlwPFPuppiJets, l1tSlwPFPuppiJetsCorrected, simEmtfDigis, simKBmtfDigis, simKBmtfStubs, simOmtfDigis, simTwinMuxDigis)
