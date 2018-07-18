import FWCore.ParameterSet.Config as cms

scoutingAlphaTVariables = cms.EDProducer("AlphaTVarProducer",
    inputJetTag = cms.InputTag("hltCaloJetIDPassed"),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
scoutingAlphaTVarAnalyzer = DQMEDAnalyzer('AlphaTVarAnalyzer',
  modulePath=cms.untracked.string("AlphaT"),
  alphaTVarCollectionName=cms.untracked.InputTag("scoutingAlphaTVariables")
  )


#this file contains the sequence for data scouting using the AlphaT analysis
scoutingAlphaTDQMSequence = cms.Sequence(
                                        scoutingAlphaTVariables*
                                        scoutingAlphaTVarAnalyzer
                                        )
