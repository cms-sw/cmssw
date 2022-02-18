import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import AODSIMEventContent
EXODisappTrkSkimContent = AODSIMEventContent.clone()

EXODisappTrkSkimContent.outputCommands.append('drop *')
EXODisappTrkSkimContent.outputCommands.append('keep *_reducedHcalRecHits_*_*')
EXODisappTrkSkimContent.outputCommands.append('keep *_reducedEcalRecHits*_*_*')

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

hltDisappTrk = copy.deepcopy(hltHighLevel)
hltDisappTrk.throw = cms.bool(False)

hltDisappTrk.HLTPaths = [
    #"HLT_MET105_IsoTrk50_v*",
    "MC_PFMET_v17"
]

hltDisappTrk.throw = False
hltDisappTrk.andOr = True

disappTrkSelection=cms.EDFilter("TrackSelector", 
    src = cms.InputTag("generalTracks"),
    cut = cms.string('pt > 25 && abs(eta()) < 2.1'),
    filter = cms.bool(True)
)

# disappTrk skim sequence
EXODisappTrkSkimSequence = cms.Sequence(
    hltDisappTrk * disappTrkSelection
    )
