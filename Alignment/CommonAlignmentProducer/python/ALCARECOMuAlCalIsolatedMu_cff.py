import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlCalIsolatedMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlCalIsolatedMu',
    throw = False # tolerate triggers not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC" 
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlCalIsolatedDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(False), # False = at least one detector from DetectorType map above is ON
    DebugOn      = cms.untracked.bool(False)
)

#________________________________Muon selection____________________________________
# AlCaReco selected muons for track based muon alignment
import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlCalIsolatedMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src    = cms.InputTag("muons"),
    filter = cms.bool(True), # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
# DT calibration needs as many muons with DIGIs as possible, which in cosmic ray runs means standAloneMuons
#    nHitMinGB = 1, # muon collections now merge globalMuons, standAlone, and trackerMuons: this stream has always assumed globalMuons
    ptMin  = cms.double(0.),
    pMin   = cms.double(20.),
    etaMin = cms.double(-2.6),
    etaMax = cms.double(2.6),
    )

seqALCARECOMuAlCalIsolatedMu = cms.Sequence(ALCARECOMuAlCalIsolatedMuHLT + ALCARECOMuAlCalIsolatedDCSFilter + ALCARECOMuAlCalIsolatedMu)
