import FWCore.ParameterSet.Config as cms

# DQM monitor module for BPhysics: onia resonances
qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
                                    photonCollection = cms.InputTag("photons"),
)



