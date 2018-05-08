import FWCore.ParameterSet.Config as cms

DQM_CPPF = cms.EDAnalyzer('DQM_CPPF',
                      cppfdigiLabel = cms.InputTag("emulatorCppfDigis","recHit")                  
)
