import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer2DEClient = cms.EDAnalyzer("L1TStage2CaloLayer2DEClient",
                  monitorDir = cms.untracked.string('L1T2016EMU/L1TStage2CaloLayer2DERatio'),
                  inputDataDir = cms.untracked.string('L1T2016/L1TStage2CaloLayer2'),
                  inputEmulDir = cms.untracked.string('L1T2016EMU/L1TStage2CaloLayer2EMU')
)


