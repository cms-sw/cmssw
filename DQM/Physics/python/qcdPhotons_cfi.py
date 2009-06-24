import FWCore.ParameterSet.Config as cms

# DQM monitor module for BPhysics: onia resonances
QcdPhotonsAnalyzer = cms.EDAnalyzer("QcdPhotons",
                                    photonCollection = cms.InputTag("photons"),
)



