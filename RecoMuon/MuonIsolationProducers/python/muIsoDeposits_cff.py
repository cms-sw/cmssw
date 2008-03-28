# The following comments couldn't be translated into the new config version:

#the same but for FastSim muons

#for bw compat (kill later)

import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
#the default set of includes
from RecoMuon.MuonIsolationProducers.muIsoDeposits_setup_cff import *
from RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi import *
from RecoMuon.MuonIsolationProducers.muIsoDepositJets_cfi import *
from RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorTowers_cfi import *
from RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorHits_cfi import *
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi import *
muParamGlobalIsoDepositCtfTk = copy.deepcopy(muIsoDepositTk)
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorTowers_cfi import *
muParamGlobalIsoDepositCalByAssociatorTowers = copy.deepcopy(muIsoDepositCalByAssociatorTowers)
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorHits_cfi import *
muParamGlobalIsoDepositCalByAssociatorHits = copy.deepcopy(muIsoDepositCalByAssociatorHits)
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositJets_cfi import *
muParamGlobalIsoDepositJets = copy.deepcopy(muIsoDepositJets)
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi import *
muParamGlobalIsoDepositGsTk = copy.deepcopy(muIsoDepositTk)
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi import *
muParamGlobalIsoDepositTk = copy.deepcopy(muIsoDepositTk)
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi import *
muParamGlobalIsoDepositCalEcal = copy.deepcopy(muIsoDepositCal)
import copy
from RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi import *
muParamGlobalIsoDepositCalHcal = copy.deepcopy(muIsoDepositCal)
#
# and now sequences of the above
#
#------------------------------
# "standard sequences"
muIsoDeposits_muons = cms.Sequence(muIsoDepositTk+muIsoDepositCalByAssociatorTowers+muIsoDepositJets)
#old one, using a reduced config set
muIsoDeposits_ParamGlobalMuonsOld = cms.Sequence(muParamGlobalIsoDepositGsTk+muParamGlobalIsoDepositCalEcal+muParamGlobalIsoDepositCalHcal)
muIsoDeposits_ParamGlobalMuons = cms.Sequence(muParamGlobalIsoDepositTk+muParamGlobalIsoDepositCalByAssociatorTowers+muParamGlobalIsoDepositJets)
muParamGlobalIsoDepositCtfTk.IOPSet = cms.PSet(
    MIsoDepositParamGlobalViewIOBlock
)
muParamGlobalIsoDepositCtfTk.ExtractorPSet = cms.PSet(
    MIsoTrackExtractorGsBlock
)
muParamGlobalIsoDepositCalByAssociatorTowers.IOPSet = cms.PSet(
    MIsoDepositParamGlobalViewMultiIOBlock
)
muParamGlobalIsoDepositCalByAssociatorHits.IOPSet = cms.PSet(
    MIsoDepositParamGlobalViewMultiIOBlock
)
muParamGlobalIsoDepositJets.IOPSet = cms.PSet(
    MIsoDepositParamGlobalViewIOBlock
)
muParamGlobalIsoDepositGsTk.IOPSet = cms.PSet(
    MIsoDepositParamGlobalIOBlock
)
muParamGlobalIsoDepositGsTk.ExtractorPSet = cms.PSet(
    MIsoTrackExtractorGsBlock
)
muParamGlobalIsoDepositTk.IOPSet = cms.PSet(
    MIsoDepositParamGlobalIOBlock
)
muParamGlobalIsoDepositTk.ExtractorPSet = cms.PSet(
    MIsoTrackExtractorBlock
)
muParamGlobalIsoDepositCalEcal.IOPSet = cms.PSet(
    MIsoDepositParamGlobalIOBlock
)
muParamGlobalIsoDepositCalEcal.ExtractorPSet = cms.PSet(
    MIsoCaloExtractorEcalBlock
)
muParamGlobalIsoDepositCalHcal.IOPSet = cms.PSet(
    MIsoDepositParamGlobalIOBlock
)
muParamGlobalIsoDepositCalHcal.ExtractorPSet = cms.PSet(
    MIsoCaloExtractorHcalBlock
)

