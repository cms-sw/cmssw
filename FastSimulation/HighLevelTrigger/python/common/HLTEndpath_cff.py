import FWCore.ParameterSet.Config as cms

import copy
from L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi import *
l1gtTrigReport = copy.deepcopy(l1GtTrigReport)
import copy
from HLTrigger.HLTanalyzers.hlTrigReport_cfi import *
hltTrigReport = copy.deepcopy(hlTrigReport)
import copy
from HLTrigger.HLTcore.triggerSummaryProducerRAW_cfi import *
triggerSummaryRAW = copy.deepcopy(triggerSummaryProducerRAW)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
triggerSummaryRAWprescaler = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTfilters.hltBool_cfi import *
boolFinal = copy.deepcopy(hltBool)
options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
triggerSummaryProducerAOD = cms.EDFilter("TriggerSummaryProducerAOD",
    TriggerSummaryAOD,
    processName = cms.string('@')
)

triggerSummaryAOD = cms.EDFilter("TriggerSummaryProducerAOD",
    processName = cms.string('@'),
    collections = cms.VInputTag(cms.InputTag("l1IsoRecoEcalCandidate"), cms.InputTag("l1NonIsoRecoEcalCandidate"), cms.InputTag("pixelMatchElectronsL1IsoForHLT"), cms.InputTag("pixelMatchElectronsL1NonIsoForHLT"), cms.InputTag("pixelMatchElectronsL1IsoLargeWindowForHLT"), 
        cms.InputTag("pixelMatchElectronsL1NonIsoLargeWindowForHLT"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltL2MuonCandidates"), cms.InputTag("MCJetCorJetIcone5"), cms.InputTag("iterativeCone5CaloJets"), 
        cms.InputTag("MCJetCorJetIcone5Regional"), cms.InputTag("iterativeCone5CaloJetsRegional"), cms.InputTag("met"), cms.InputTag("htMet"), cms.InputTag("htMetIC5"), 
        cms.InputTag("isolatedL3SingleTau"), cms.InputTag("isolatedL3SingleTauMET"), cms.InputTag("isolatedL25PixelTau"), cms.InputTag("hltBLifetimeL3Jets"), cms.InputTag("hltBSoftmuonL25Jets"), 
        cms.InputTag("hltMuTracks"), cms.InputTag("hltMuTracks"), cms.InputTag("hltMumukAllConeTracks"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau"), 
        cms.InputTag("l1IsoRecoEcalCandidate"), cms.InputTag("pixelMatchElectronsL1IsoForHLT"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorElectronTau"), cms.InputTag("hltIsolatedTauJetsSelectorL3ElectronTau"), cms.InputTag("hltL3MuonCandidates"), 
        cms.InputTag("hltBSoftmuonL3BJetTags"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltBLifetimeL3BJetTags"), cms.InputTag("l1IsoRecoEcalCandidate"), cms.InputTag("l1NonIsoRecoEcalCandidate"), 
        cms.InputTag("pixelMatchElectronsL1IsoForHLT"), cms.InputTag("pixelMatchElectronsL1NonIsoForHLT"), cms.InputTag("hltBLifetimeL3BJetTags"), cms.InputTag("l1IsoRecoEcalCandidate"), cms.InputTag("l1NonIsoRecoEcalCandidate"), 
        cms.InputTag("pixelMatchElectronsL1IsoForHLT"), cms.InputTag("pixelMatchElectronsL1NonIsoForHLT"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltL2MuonCandidates"), cms.InputTag("hltL3MuonCandidates"), 
        cms.InputTag("MCJetCorJetIcone5"), cms.InputTag("iterativeCone5CaloJets"), cms.InputTag("MCJetCorJetIcone5Regional"), cms.InputTag("iterativeCone5CaloJetsRegional"), cms.InputTag("isolPixelTrackProd")),
    filters = cms.VInputTag(cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1IsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter"), cms.InputTag("hltL1IsoDoubleExclPhotonTrackIsolFilter"), cms.InputTag("hltL1IsoSinglePhotonPrescaledTrackIsolFilter"), cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleExclElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleElectronZeePMMassFilter"), cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("SingleMuIsoL3IsoFiltered"), cms.InputTag("SingleMuNoIsoL3PreFiltered"), 
        cms.InputTag("DiMuonIsoL3IsoFiltered"), cms.InputTag("DiMuonNoIsoL3PreFiltered"), cms.InputTag("ZMML3Filtered"), cms.InputTag("JpsiMML3Filtered"), cms.InputTag("UpsilonMML3Filtered"), 
        cms.InputTag("multiMuonNoIsoL3PreFiltered"), cms.InputTag("SameSignMuL3IsoFiltered"), cms.InputTag("ExclDiMuonIsoL3IsoFiltered"), cms.InputTag("SingleMuPrescale3L3PreFiltered"), cms.InputTag("SingleMuPrescale5L3PreFiltered"), 
        cms.InputTag("SingleMuPrescale710L3PreFiltered"), cms.InputTag("SingleMuPrescale77L3PreFiltered"), cms.InputTag("DiMuonNoIsoL3PreFilteredRelaxedVtx2cm"), cms.InputTag("DiMuonNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("SingleMuNoIsoL3PreFilteredRelaxedVtx2cm"), 
        cms.InputTag("SingleMuNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("SingleMuStartupL2PreFiltered"), cms.InputTag("hlt1jet200"), cms.InputTag("hlt1jet150"), cms.InputTag("hlt1jet110"), 
        cms.InputTag("hlt1jet60"), cms.InputTag("hlt1jet30"), cms.InputTag("hlt2jet150"), cms.InputTag("hlt3jet85"), cms.InputTag("hlt4jet60"), 
        cms.InputTag("hlt1MET65"), cms.InputTag("hlt1MET55"), cms.InputTag("hlt1MET30"), cms.InputTag("hlt1MET20"), cms.InputTag("hlt2jetAco"), 
        cms.InputTag("hlt1jet180"), cms.InputTag("hlt2jet125"), cms.InputTag("hlt3jet60"), cms.InputTag("hlt4jet35"), cms.InputTag("hlt1jet1METAco"), 
        cms.InputTag("hlt2jetvbf"), cms.InputTag("hltnv"), cms.InputTag("hltPhi2metAco"), cms.InputTag("hltPhiJet1metAco"), cms.InputTag("hltPhiJet2metAco"), 
        cms.InputTag("hltPhiJet1Jet2Aco"), cms.InputTag("hlt1SumET120"), cms.InputTag("hlt1HT400"), cms.InputTag("hlt1HT350"), cms.InputTag("hltRapGap"), 
        cms.InputTag("hltdijetave110"), cms.InputTag("hltdijetave150"), cms.InputTag("hltdijetave200"), cms.InputTag("hltdijetave30"), cms.InputTag("hltdijetave60"), 
        cms.InputTag("filterL3SingleTau"), cms.InputTag("filterL3SingleTauMET"), cms.InputTag("filterL25PixelTau"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltBSoftmuonL3filter"), 
        cms.InputTag("hltBSoftmuonByDRL3filter"), cms.InputTag("displacedJpsitoMumuFilter"), cms.InputTag("hltmmkFilter"), cms.InputTag("hltMuonTauIsoL3IsoFiltered"), cms.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTau"), 
        cms.InputTag("hltElectronTrackIsolFilterElectronTau"), cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau"), cms.InputTag("hltFilterIsolatedTauJetsL3ElectronTau"), cms.InputTag("hltFilterPixelTrackIsolatedTauJetsElectronTau"), cms.InputTag("hltBSoftmuonL3filter"), 
        cms.InputTag("MuBIsoL3IsoFiltered"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("MuBIsoL3IsoFiltered"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("ElBElectronTrackIsolFilter"), 
        cms.InputTag("hltemuL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltemuNonIsoL1IsoTrackIsolFilter"), cms.InputTag("EMuL3MuonIsoFilter"), cms.InputTag("hltNonIsoEMuL3MuonPreFilter"), cms.InputTag("MuJetsL3IsoFiltered"), 
        cms.InputTag("MuJetsHLT1jet40"), cms.InputTag("MuNoL2IsoJetsL3IsoFiltered"), cms.InputTag("MuNoL2IsoJetsHLT1jet40"), cms.InputTag("MuNoIsoJetsL3PreFiltered"), cms.InputTag("MuNoIsoJetsHLT1jet50"), 
        cms.InputTag("isolPixelTrackFilter"))
)

HLTEndpath1 = cms.EndPath(l1gtTrigReport+hltTrigReport)
triggerFinalPath = cms.Sequence(triggerSummaryAOD+triggerSummaryRAWprescaler+triggerSummaryRAW+boolFinal)
boolFinal.result = False

