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
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX',
                               'DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi
ALCARECOTkAlUpsilonMuMuGoodMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlGoodIdMuonSelector.clone()
ALCARECOTkAlUpsilonMuMuRelCombIsoMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlRelCombIsoMuonSelector.clone(
    src = 'ALCARECOTkAlUpsilonMuMuGoodMuons',
    cut = '(isolationR03().sumPt + isolationR03().emEt + isolationR03().hadEt)/pt  < 0.3'

)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlUpsilonMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlUpsilonMuMu.filter = True ##do not store empty events

ALCARECOTkAlUpsilonMuMu.applyBasicCuts = True
ALCARECOTkAlUpsilonMuMu.ptMin = 3. ##GeV
ALCARECOTkAlUpsilonMuMu.etaMin = -3.5
ALCARECOTkAlUpsilonMuMu.etaMax = 3.5
ALCARECOTkAlUpsilonMuMu.nHitMin = 0

ALCARECOTkAlUpsilonMuMu.GlobalSelector.muonSource = 'ALCARECOTkAlUpsilonMuMuRelCombIsoMuons'
# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
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
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.numberOfCandidates = 1	 

seqALCARECOTkAlUpsilonMuMu = cms.Sequence(ALCARECOTkAlUpsilonMuMuHLT+ALCARECOTkAlUpsilonMuMuDCSFilter+ALCARECOTkAlUpsilonMuMuGoodMuons+ALCARECOTkAlUpsilonMuMuRelCombIsoMuons+ALCARECOTkAlUpsilonMuMu)

## customizations for the pp_on_AA eras
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(ALCARECOTkAlUpsilonMuMuHLT,
                                      eventSetupPathsKey='TkAlUpsilonMuMuHI'
)
