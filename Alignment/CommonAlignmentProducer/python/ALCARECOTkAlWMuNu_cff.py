# AlCaReco for track based alignment using WMuNu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlWMuNuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = ['HLT_IsoMu11'], ##these need further studies
    throw = False # tolerate triggers stated above, but not available
)

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlWMuNuDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX',
                               'DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi
ALCARECOTkAlWMuNuGoodMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlGoodIdMuonSelector.clone()
ALCARECOTkAlWMuNuRelCombIsoMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlRelCombIsoMuonSelector.clone(
    src = 'ALCARECOTkAlWMuNuGoodMuons'
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlWMuNu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOTkAlWMuNu.filter = True ##do not store empty events

ALCARECOTkAlWMuNu.applyBasicCuts = True
ALCARECOTkAlWMuNu.ptMin = 20.0 ##GeV

ALCARECOTkAlWMuNu.etaMin = -3.5
ALCARECOTkAlWMuNu.etaMax = 3.5
ALCARECOTkAlWMuNu.nHitMin = 0

ALCARECOTkAlWMuNu.GlobalSelector.muonSource = 'ALCARECOTkAlWMuNuRelCombIsoMuons'
# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
ALCARECOTkAlWMuNu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlWMuNu.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlWMuNu.GlobalSelector.applyJetCountFilter = True
ALCARECOTkAlWMuNu.GlobalSelector.minJetPt = 40 ##GeV
ALCARECOTkAlWMuNu.GlobalSelector.maxJetCount = 3

ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyMissingETFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.minXMass = 40.0 ##GeV
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.maxXMass = 200.0 ##GeV
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.charge = 1
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.useUnsignedCharge = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyAcoplanarityFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.acoplanarDistance = 1 ##radian

seqALCARECOTkAlWMuNu = cms.Sequence(ALCARECOTkAlWMuNuHLT+ALCARECOTkAlWMuNuDCSFilter+ALCARECOTkAlWMuNuGoodMuons+ALCARECOTkAlWMuNuRelCombIsoMuons+ALCARECOTkAlWMuNu)
