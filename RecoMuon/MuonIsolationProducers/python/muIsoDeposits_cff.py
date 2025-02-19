# The following comments couldn't be translated into the new config version:

#the same but for FastSim muons

#for bw compat (kill later)

import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
#the default set of includes
from RecoMuon.MuonIsolationProducers.muIsoDeposits_setup_cff import *
#the default set of modules first
#from RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi import *
#from RecoMuon.MuonIsolationProducers.muIsoDepositJets_cfi import *
#from RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorTowers_cfi import *
from RecoMuon.MuonIsolationProducers.muIsoDepositCopies_cfi import *
from RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorHits_cfi import *
import RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi
muParamGlobalIsoDepositCtfTk = RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi.muIsoDepositTk.clone()
import RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorTowers_cfi
muParamGlobalIsoDepositCalByAssociatorTowers = RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorTowers_cfi.muIsoDepositCalByAssociatorTowers.clone()
import RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorHits_cfi
muParamGlobalIsoDepositCalByAssociatorHits = RecoMuon.MuonIsolationProducers.muIsoDepositCalByAssociatorHits_cfi.muIsoDepositCalByAssociatorHits.clone()
import RecoMuon.MuonIsolationProducers.muIsoDepositJets_cfi
muParamGlobalIsoDepositJets = RecoMuon.MuonIsolationProducers.muIsoDepositJets_cfi.muIsoDepositJets.clone()
import RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi
muParamGlobalIsoDepositGsTk = RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi.muIsoDepositTk.clone()
import RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi
muParamGlobalIsoDepositTk = RecoMuon.MuonIsolationProducers.muIsoDepositTk_cfi.muIsoDepositTk.clone()
import RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi
muParamGlobalIsoDepositCalEcal = RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi.muIsoDepositCal.clone()
import RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi
muParamGlobalIsoDepositCalHcal = RecoMuon.MuonIsolationProducers.muIsoDepositCal_cfi.muIsoDepositCal.clone()
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


