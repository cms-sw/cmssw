import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SMPDQM= DQMEDAnalyzer('SMPDQM',
                      pvs = cms.InputTag('offlinePrimaryVertices'),
                      jets  = cms.InputTag("ak4PFJetsCHS"),
                      mets = cms.VInputTag("pfMet"),
                      elecCollection = cms.InputTag('gedGsfElectrons'),
                      muonCollection = cms.InputTag('muons'),
                      pfMETCollection          = cms.InputTag("pfMet"),
                      )
