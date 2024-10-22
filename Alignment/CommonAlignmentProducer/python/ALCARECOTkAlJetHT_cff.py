# AlCaReco for track based alignment using min. bias events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlJetHTHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlJetHTHLT',
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlJetHTDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import FWCore.Modules.preScaler_cfi
ALCARECOTkAlJetHTPrescaler = FWCore.Modules.preScaler_cfi.preScaler.clone(
    prescaleFactor = cms.int32(10), # selects one event out of 10
    prescaleOffset = cms.int32(1)
)
    
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlJetHT = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    filter = True, ##do not store empty events	
    applyBasicCuts = True,
    ptMin = 0.65, ##GeV
    pMin = 1.5, ##GeV
    etaMin = -3.5,
    etaMax = 3.5,
    nHitMin = 7 ## at least 7 hits required
)

ALCARECOTkAlJetHT.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlJetHT.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlJetHT.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlJetHT.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlJetHT.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlJetHT = cms.Sequence(ALCARECOTkAlJetHTHLT+ALCARECOTkAlJetHTDCSFilter+ALCARECOTkAlJetHTPrescaler*ALCARECOTkAlJetHT)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(ALCARECOTkAlJetHT, etaMin = -4, etaMax = 4)
