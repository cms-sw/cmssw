# AlCaReco for track based alignment using min. bias events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
#Note the MinBias selection should contain as many tracks as possible but no overlaps. So the HLT selection selects any event that is not selected in another TkAl* selector.
ALCARECOTkAlMinBiasNOTHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlMinBiasNOT',
    throw = False # tolerate triggers stated above, but not available
    )
ALCARECOTkAlMinBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlMinBias',
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlMinBiasDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMinBias = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlMinBias.filter = True ##do not store empty events	

ALCARECOTkAlMinBias.applyBasicCuts = True
ALCARECOTkAlMinBias.ptMin = 0.65 ##GeV
ALCARECOTkAlMinBias.pMin = 1.5 ##GeV


ALCARECOTkAlMinBias.etaMin = -3.5
ALCARECOTkAlMinBias.etaMax = 3.5
ALCARECOTkAlMinBias.nHitMin = 7 ## at least 7 hits required
ALCARECOTkAlMinBias.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMinBias.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlMinBias = cms.Sequence(ALCARECOTkAlMinBiasHLT*~ALCARECOTkAlMinBiasNOTHLT+ALCARECOTkAlMinBiasDCSFilter+ALCARECOTkAlMinBias)

## customizations for the pp_on_AA eras
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
(pp_on_XeXe_2017 | pp_on_AA_2018).toModify(ALCARECOTkAlMinBiasHLT,
                                           eventSetupPathsKey='TkAlMinBiasHI'
                                           )

(pp_on_XeXe_2017 | pp_on_AA_2018).toModify(ALCARECOTkAlMinBias,
                                           trackQualities = cms.vstring("highPurity")
                                           )

