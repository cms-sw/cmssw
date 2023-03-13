import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt

from Configuration.EventContent.EventContent_cff import AODSIMEventContent
EXODisappMuonSkimContent = AODSIMEventContent.clone()
EXODisappMuonSkimContent.outputCommands.append('keep *_hbhereco_*_*')
EXODisappMuonSkimContent.outputCommands.append('keep *_horeco_*_*')
EXODisappMuonSkimContent.outputCommands.append('keep *_csc2DRecHits_*_*')

exoDisappMuonsHLT = hlt.hltHighLevel.clone(
   throw = False,
   andOr = True,
   HLTPaths = [
	"HLT_IsoMu*_v*"
   ]
)

from Configuration.Skimming.disappearingMuonsSkimming_cfi import *
disappMuonsSelection = disappearingMuonsSkimming.clone()

EXODisappMuonSkimSequence = cms.Sequence(
    exoDisappMuonsHLT+disappMuonsSelection
)
