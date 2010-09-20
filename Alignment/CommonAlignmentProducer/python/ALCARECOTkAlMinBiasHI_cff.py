# AlCaReco for track based alignment using min. bias events in heavy ion data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import *

import HLTrigger.HLTfilters.hltHighLevel_cfi
# Note the MinBias selection should contain as many tracks as possible but no overlaps.
# So the HLT selection selects any event that is not selected in another TkAl* selector.
ALCARECOTkAlMinBiasHINOTHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlMinBiasHINOT',
    throw = False # tolerate triggers stated above, but not available
    )
ALCARECOTkAlMinBiasHIHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlMinBiasHI',
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlMinBiasHIDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMinBiasHI = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlMinBiasHI.src = 'hiSelectedTracks'
ALCARECOTkAlMinBiasHI.filter = True ##do not store empty events

ALCARECOTkAlMinBiasHI.applyBasicCuts = True
ALCARECOTkAlMinBiasHI.ptMin = 0.65 ##GeV
ALCARECOTkAlMinBiasHI.pMin = 1.5 ##GeV


ALCARECOTkAlMinBiasHI.etaMin = -3.5
ALCARECOTkAlMinBiasHI.etaMax = 3.5
ALCARECOTkAlMinBiasHI.nHitMin = 7 ## at least 7 hits required
ALCARECOTkAlMinBiasHI.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMinBiasHI.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlMinBiasHI.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMinBiasHI.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMinBiasHI.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlMinBiasHI = cms.Sequence(ALCARECOTkAlMinBiasHIHLT*~ALCARECOTkAlMinBiasHINOTHLT+ALCARECOTkAlMinBiasHIDCSFilter+ALCARECOTkAlMinBiasHI)
