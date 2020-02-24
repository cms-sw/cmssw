import FWCore.ParameterSet.Config as cms

demo = cms.EDAnalyzer('PhotonAnalyzer',
                      photonToken                          =  cms.InputTag("slimmedPhotons"),
)
