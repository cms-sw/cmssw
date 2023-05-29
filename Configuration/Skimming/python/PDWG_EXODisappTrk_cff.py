import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import AODSIMEventContent
EXODisappTrkSkimContent = AODSIMEventContent.clone()
EXODisappTrkSkimContent.outputCommands.append('drop *')
EXODisappTrkSkimContent.outputCommands.append('keep *_reducedHcalRecHits_*_*')
EXODisappTrkSkimContent.outputCommands.append('keep *_reducedEcalRecHits*_*_*')

import HLTrigger.HLTfilters.hltHighLevel_cfi as _hltHighLevel
hltDisappTrk = _hltHighLevel.hltHighLevel.clone(
   throw = False,
   andOr = True,
   HLTPaths = [
      "HLT_PFMET*_PFMHT*_IDTight_v*",
      "HLT_PFMETTypeOne*_PFMHT*_IDTight_v*",
      "HLT_PFMETNoMu*_PFMHTNoMu*_IDTight_v*",
      "HLT_MET*_IsoTrk*_v*",
      "HLT_PFMET*_*Cleaned_v*",
      "HLT_Ele*_WPTight_Gsf_v*",
      "HLT_Ele*_WPLoose_Gsf_v*",
      "HLT_IsoMu*_v*",
      "HLT_MediumChargedIsoPFTau*HighPtRelaxedIso_Trk50_eta2p1_v*",
      "HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v*",
      "HLT_DoubleMediumDeepTauPFTauHPS*_L2NN_eta2p1_*",
      "HLT_LooseDeepTauPFTauHPS*_L2NN_eta2p1_v*"
   ]
)

disappTrkSelection=cms.EDFilter("TrackSelector", 
    src = cms.InputTag("generalTracks"),
    cut = cms.string('pt > 25 && abs(eta()) < 2.1'),
    filter = cms.bool(True)
)

# disappTrk skim sequence
EXODisappTrkSkimSequence = cms.Sequence(
    hltDisappTrk * disappTrkSelection
    )
