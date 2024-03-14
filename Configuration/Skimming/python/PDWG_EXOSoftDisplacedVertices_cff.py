import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import AODSIMEventContent
EXOSoftDisplacedVerticesSkimContent = AODSIMEventContent.clone()

import HLTrigger.HLTfilters.hltHighLevel_cfi as _hltHighLevel

hltSoftDV = _hltHighLevel.hltHighLevel.clone(
   throw = False,
   andOr = True,
   HLTPaths = [
    "HLT_PFHT*_PFMET*_PFMHT*_IDTight_v*",
    "HLT_PFMET*_PFMHT*_IDTight_v*",
    "HLT_PFMET*_PFMHT*_IDTight_PFHT*_v*",
    "HLT_PFMETTypeOne*_PFMHT*_IDTight_v*",
    "HLT_PFMETTypeOne*_PFMHT*_IDTight_PFHT*_v*",
    "HLT_PFMETNoMu*_PFMHTNoMu*_IDTight_v*",
    "HLT_PFMETNoMu*_PFMHTNoMu*_IDTight_PFHT*_v*",
    "HLT_MonoCentralPFJet*_PFMETNoMu*_PFMHTNoMu*_IDTight_v*",
    "HLT_PFMET*_*Cleaned_v*"
   ]
)

softDVSelection=cms.EDFilter("CandViewSelector",
    src = cms.InputTag("pfMet"),
    cut = cms.string( "pt()>140" ),
    filter = cms.bool(True)
)

EXOSoftDisplacedVerticesSkimSequence = cms.Sequence(
    hltSoftDV * softDVSelection
    )