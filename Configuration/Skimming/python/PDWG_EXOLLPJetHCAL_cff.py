import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi as _hltHighLevel
hltLLPJetHCAL = _hltHighLevel.hltHighLevel.clone(
   throw = False,
   andOr = True,
   HLTPaths = [
      "HLT*_L1SingleLLPJet_*",
   ]
)

# disappTrk skim sequence
EXOLLPJetHCALSkimSequence = cms.Sequence(
    hltLLPJetHCAL
    )