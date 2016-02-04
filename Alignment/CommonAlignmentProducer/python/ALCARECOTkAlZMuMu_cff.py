import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for track based alignment using ZMuMu events
ALCARECOTkAlZMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlZMuMu',
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlZMuMuDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

ALCARECOTkAlZMuMuGoodMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon = 1'),
    filter = cms.bool(True)                                
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlZMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlZMuMu.filter = True ##do not store empty events

ALCARECOTkAlZMuMu.applyBasicCuts = True
ALCARECOTkAlZMuMu.ptMin = 15.0 ##GeV
ALCARECOTkAlZMuMu.etaMin = -3.5
ALCARECOTkAlZMuMu.etaMax = 3.5
ALCARECOTkAlZMuMu.nHitMin = 0

ALCARECOTkAlZMuMu.GlobalSelector.muonSource = 'ALCARECOTkAlZMuMuGoodMuonSelector'
ALCARECOTkAlZMuMu.GlobalSelector.applyIsolationtest = True
ALCARECOTkAlZMuMu.GlobalSelector.applyGlobalMuonFilter = True

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.minXMass = 65.0 ##GeV
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.maxXMass = 115.0 ##GeV
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlZMuMu = cms.Sequence(ALCARECOTkAlZMuMuHLT+ALCARECOTkAlZMuMuDCSFilter+ALCARECOTkAlZMuMuGoodMuonSelector+ALCARECOTkAlZMuMu)
