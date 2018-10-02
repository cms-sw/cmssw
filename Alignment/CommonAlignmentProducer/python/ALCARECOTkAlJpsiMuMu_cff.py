# AlCaReco for track based alignment using J/Psi->MuMu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlJpsiMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlJpsiMuMu',
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlJpsiMuMuDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX',
                               'DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi
ALCARECOTkAlJpsiMuMuGoodMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlGoodIdMuonSelector.clone()

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlJpsiMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlJpsiMuMu.filter = True ##do not store empty events

ALCARECOTkAlJpsiMuMu.applyBasicCuts = True
ALCARECOTkAlJpsiMuMu.ptMin = 0.8 ##GeV
ALCARECOTkAlJpsiMuMu.etaMin = -3.5
ALCARECOTkAlJpsiMuMu.etaMax = 3.5
ALCARECOTkAlJpsiMuMu.nHitMin = 0

ALCARECOTkAlJpsiMuMu.GlobalSelector.muonSource = 'ALCARECOTkAlJpsiMuMuGoodMuons'
# To not loose non-prompt J/Psi, do not apply any isolation
ALCARECOTkAlJpsiMuMu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlJpsiMuMu.GlobalSelector.applyGlobalMuonFilter = True

ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.minXMass = 2.7 ##GeV
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.maxXMass = 3.4 ##GeV
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.acoplanarDistance = 1 ##radian
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.numberOfCandidates = 1 	 

seqALCARECOTkAlJpsiMuMu = cms.Sequence(ALCARECOTkAlJpsiMuMuHLT+ALCARECOTkAlJpsiMuMuDCSFilter+ALCARECOTkAlJpsiMuMuGoodMuons+ALCARECOTkAlJpsiMuMu)

## customizations for the pp_on_AA eras
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
(pp_on_XeXe_2017 | pp_on_AA_2018).toModify(ALCARECOTkAlJpsiMuMuHLT,
                                           eventSetupPathsKey='TkAlJpsiMuMuHI'
                                           )
