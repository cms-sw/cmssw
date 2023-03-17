import FWCore.ParameterSet.Config as cms

#from Configuration.EventContent.EventContent_cff import AODSIMEventContent
#from Configuration.EventContent.EventContent_cff import RAWAODEventContent
#EXOLLPJetHCALSkimContent = RAWAODEventContent.clone()
#EXOLLPJetHCALSkimContent.outputCommands.append('drop *')
#EXOLLPJetHCALSkimContent.outputCommands.append('keep *_reducedHcalRecHits_*_*')

import HLTrigger.HLTfilters.hltHighLevel_cfi as _hltHighLevel
hltLLPJetHCAL = _hltHighLevel.hltHighLevel.clone(
   throw = False,
   andOr = True,
   HLTPaths = [
      "HLT_HT*_L1SingleLLPJet_*",
   ]
)

# disappTrk skim sequence
EXOLLPJetHCALSkimSequence = cms.Sequence(
    hltLLPJetHCAL #* disappTrkSelection
    )

"""
from Configuration.Skimming.PDWG_EXOLLPJetHCAL_cff import *
EXOLLPJetHCALPath = cms.Path(EXOLLPJetHCALSkimSequence)
SKIMStreamEXOLLPJetHCAL = cms.FilteredStream(
    responsible = 'PDWG', 
    name = 'EXOLLPJetHCAL', 
    paths = (EXOLLPJetHCALPath),
    content = skimRawAODContent.outputCommands+['keep *_reducedHcalRecHits_*_*'],
    selectEvents = cms.untracked.PSet(), 
    dataTier = cms.untracked.string('AOD')
    )
"""