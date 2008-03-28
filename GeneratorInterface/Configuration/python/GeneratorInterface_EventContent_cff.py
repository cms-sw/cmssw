import FWCore.ParameterSet.Config as cms

#Full Event content 
GeneratorInterfaceFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*')
)
#RECO content
GeneratorInterfaceRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*')
)
#AOD content
GeneratorInterfaceAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmGenInfoProduct_source_*_*', 'keep recoGenParticles_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*')
)

