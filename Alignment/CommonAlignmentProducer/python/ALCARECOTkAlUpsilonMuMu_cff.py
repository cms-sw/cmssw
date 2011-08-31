# AlCaReco for track based alignment using Upsilon->MuMu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlUpsilonMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlUpsilonMuMu',
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlUpsilonMuMuDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

ALCARECOTkAlUpsilonMuMuGoodMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon = 1'),
    filter = cms.bool(True)                                
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlUpsilonMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlUpsilonMuMu.filter = True ##do not store empty events

ALCARECOTkAlUpsilonMuMu.applyBasicCuts = True
ALCARECOTkAlUpsilonMuMu.ptMin = 0.8 ##GeV
ALCARECOTkAlUpsilonMuMu.etaMin = -3.5
ALCARECOTkAlUpsilonMuMu.etaMax = 3.5
ALCARECOTkAlUpsilonMuMu.nHitMin = 0

ALCARECOTkAlUpsilonMuMu.GlobalSelector.muonSource = 'ALCARECOTkAlUpsilonMuMuGoodMuonSelector'
ALCARECOTkAlUpsilonMuMu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlUpsilonMuMu.GlobalSelector.applyGlobalMuonFilter = True

ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.minXMass = 8.9 ##GeV
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.maxXMass = 9.9 ##GeV
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.acoplanarDistance = 1 ##radian

seqALCARECOTkAlUpsilonMuMu = cms.Sequence(ALCARECOTkAlUpsilonMuMuHLT+ALCARECOTkAlUpsilonMuMuDCSFilter+ALCARECOTkAlUpsilonMuMuGoodMuonSelector+ALCARECOTkAlUpsilonMuMu)
