# AlCaReco for track based alignment using isolated muon tracks - relaxed cuts for pA collisions
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlMuonIsolatedPAHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlMuonIsolatedPA',
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlMuonIsolatedPADCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX',
                               'DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMuonIsolatedPA = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    filter = True, ##do not store empty events
    applyBasicCuts = True,
    ptMin = 2.0, ##GeV 
    etaMin = -3.5,
    etaMax = 3.5,
    nHitMin = 0
)

# isolation cuts are relaxed compared to pp version
ALCARECOTkAlMuonIsolatedPA.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMuonIsolatedPA.GlobalSelector.minJetDeltaR = 0. # pp version has 0.1
ALCARECOTkAlMuonIsolatedPA.GlobalSelector.applyGlobalMuonFilter = True

ALCARECOTkAlMuonIsolatedPA.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMuonIsolatedPA.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMuonIsolatedPA.TwoBodyDecaySelector.applyAcoplanarityFilter = False

# also here no furtehr cuts on muon collection as in pp version
seqALCARECOTkAlMuonIsolatedPA = cms.Sequence(ALCARECOTkAlMuonIsolatedPAHLT
                                             +ALCARECOTkAlMuonIsolatedPADCSFilter
                                             +ALCARECOTkAlMuonIsolatedPA)
