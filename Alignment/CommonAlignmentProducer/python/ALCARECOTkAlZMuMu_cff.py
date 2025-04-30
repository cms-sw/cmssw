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
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX',
                               'DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

## standard muon selection
from Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi import *

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlZMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlZMuMu.filter = True ##do not store empty events

ALCARECOTkAlZMuMu.applyBasicCuts = True
ALCARECOTkAlZMuMu.ptMin = 15.0 ##GeV
ALCARECOTkAlZMuMu.etaMin = -3.5
ALCARECOTkAlZMuMu.etaMax = 3.5
ALCARECOTkAlZMuMu.nHitMin = 0

ALCARECOTkAlZMuMu.GlobalSelector.muonSource = 'TkAlRelCombIsoMuonSelector'
# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
ALCARECOTkAlZMuMu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlZMuMu.GlobalSelector.applyGlobalMuonFilter = True

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.minXMass = 65.0 ##GeV
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.maxXMass = 115.0 ##GeV
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.numberOfCandidates = 1

## for the GEN level information
TkAlZMuMuGenMuonSelector = cms.EDFilter("GenParticleSelector",
                                        src = cms.InputTag("genParticles"),
                                        cut = cms.string("abs(pdgId) == 13"), # Select only muons
                                        filter = cms.bool(False),
                                        throwOnMissing = cms.untracked.bool(False))

seqALCARECOTkAlZMuMu = cms.Sequence(ALCARECOTkAlZMuMuHLT+
                                    ALCARECOTkAlZMuMuDCSFilter+
                                    seqALCARECOTkAlRelCombIsoMuons+
                                    ALCARECOTkAlZMuMu+
                                    TkAlZMuMuGenMuonSelector)

## customizations for the pp_on_AA eras
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(ALCARECOTkAlZMuMuHLT,
                                      eventSetupPathsKey='TkAlZMuMuHI'
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(ALCARECOTkAlZMuMu, etaMin = -4, etaMax = 4)
-- dummy change --
