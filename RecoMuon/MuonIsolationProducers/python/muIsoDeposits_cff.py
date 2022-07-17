# The following comments couldn't be translated into the new config version:

#the same but for FastSim muons

#for bw compat (kill later)

import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
#the default set of includes
from RecoMuon.MuonIsolationProducers.muIsoDeposits_setup_cff import *
#the default set of modules first
from RecoMuon.MuonIsolationProducers.muIsoDepositCopies_cfi import *
from RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorHits_cfi import *
import RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi
import RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorTowers_cfi
import RecoMuon.MuonIsolationProducers.muIsoDepositJets_cfi
import RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi

muParamGlobalIsoDepositCtfTk   = RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi.muIsoDepositTk.clone()
muParamGlobalIsoDepositJets    = RecoMuon.MuonIsolationProducers.muIsoDepositJets_cfi.muIsoDepositJets.clone()
muParamGlobalIsoDepositGsTk    = RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi.muIsoDepositTk.clone()
muParamGlobalIsoDepositTk      = RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi.muIsoDepositTk.clone()
muParamGlobalIsoDepositCalEcal = RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi.muIsoDepositCal.clone()
muParamGlobalIsoDepositCalHcal = RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi.muIsoDepositCal.clone()
muParamGlobalIsoDepositCalByAssociatorTowers = RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorTowers_cfi.muIsoDepositCalByAssociatorTowers.clone()
muParamGlobalIsoDepositCalByAssociatorHits   = RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorHits_cfi.muIsoDepositCalByAssociatorHits.clone()
#
# and now sequences of the above
#
#------------------------------
# "standard sequences"
muIsoDeposits_muonsTask = cms.Task(muIsoDepositTk,muIsoDepositCalByAssociatorTowers,muIsoDepositJets)
muIsoDeposits_muons = cms.Sequence(muIsoDeposits_muonsTask)
# "displaced sequences"
muIsoDeposits_displacedMuonsTask = cms.Task(muIsoDepositTkDisplaced,muIsoDepositCalByAssociatorTowersDisplaced,muIsoDepositJetsDisplaced)
muIsoDeposits_displacedMuons = cms.Sequence(muIsoDeposits_displacedMuonsTask)
#old one, using a reduced config set
muIsoDeposits_ParamGlobalMuonsOldTask = cms.Task(muParamGlobalIsoDepositGsTk,muParamGlobalIsoDepositCalEcal,muParamGlobalIsoDepositCalHcal)
muIsoDeposits_ParamGlobalMuonsOld = cms.Sequence(muIsoDeposits_ParamGlobalMuonsOldTask)
muIsoDeposits_ParamGlobalMuonsTask = cms.Task(muParamGlobalIsoDepositTk,muParamGlobalIsoDepositCalByAssociatorTowers,muParamGlobalIsoDepositJets)
muIsoDeposits_ParamGlobalMuons = cms.Sequence(muIsoDeposits_ParamGlobalMuonsTask)
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
