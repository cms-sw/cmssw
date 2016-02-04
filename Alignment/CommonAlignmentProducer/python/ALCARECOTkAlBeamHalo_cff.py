# AlCaReco for track based alignment using min. bias events
import FWCore.ParameterSet.Config as cms

#  module ALCARECOTkAlBeamHaloHLT = hltHighLevel from "HLTrigger/HLTfilters/data/hltHighLevel.cfi"
#  replace ALCARECOTkAlBeamHaloHLT.andOr = true # choose logical OR between Triggerbits
# which is the BeamHalo HLT Tag?
#  replace ALCARECOTkAlBeamHaloHLT.HLTPaths = {""}
#  replace ALCARECOTkAlBeamHaloHLT.throw = false

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlBeamHaloDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlBeamHalo = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOTkAlBeamHalo.src = 'beamhaloTracks'
ALCARECOTkAlBeamHalo.filter = True ##do not store empty events

ALCARECOTkAlBeamHalo.applyBasicCuts = True
ALCARECOTkAlBeamHalo.ptMin = 0.0 ##GeV

ALCARECOTkAlBeamHalo.etaMin = -9999
ALCARECOTkAlBeamHalo.etaMax = 9999
ALCARECOTkAlBeamHalo.nHitMin = 3
ALCARECOTkAlBeamHalo.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlBeamHalo.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlBeamHalo.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlBeamHalo.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlBeamHalo.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlBeamHalo = cms.Sequence(ALCARECOTkAlBeamHaloDCSFilter+ALCARECOTkAlBeamHalo)
