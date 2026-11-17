import FWCore.ParameterSet.Config as cms

from ..modules.hltDoubleTkMuonUpsilonL1TkMuonFilter_cfi import *
from ..modules.hltL3fL1DoubleTkMuonUpsilonPreFiltered0_cfi import *
from ..modules.hltDimuon0UpsilonL3Filtered_cfi import *
from ..modules.hltDisplacedmumuVtxProducerDimuon0Upsilon_cfi import *
from ..modules.hltDisplacedmumuFilterDimuon0Upsilon_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTMuonlocalrecoSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

# Port of the Run-3 HLT_Dimuon0_Upsilon path (HLT_GRun): L1 gate on the opposite-sign double-muon
# seeds, Phase-2 L3 muon reconstruction (as in HLT_Mu37_Mu27_FromL1TkMuon), two L3 muons matched to
# the L1 seed muons, 8 < m(mumu) < 12 GeV and a dimuon vertex with probability > 0.005.
HLT_Dimuon0_Upsilon_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + hltDoubleTkMuonUpsilonL1TkMuonFilter
    + HLTMuonlocalrecoSequence
    + HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + HLTMuonsSequence
    + hltPhase2L3MuonCandidates
    + hltL3fL1DoubleTkMuonUpsilonPreFiltered0
    + hltDimuon0UpsilonL3Filtered
    + hltDisplacedmumuVtxProducerDimuon0Upsilon
    + hltDisplacedmumuFilterDimuon0Upsilon
    + HLTEndSequence
)
