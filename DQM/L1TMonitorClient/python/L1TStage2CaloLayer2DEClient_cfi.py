import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer2DEClient = cms.EDAnalyzer("L1TStage2CaloLayer2Client",
                  monitorDir = cms.untracked.string('L1T2016EMU/L1TStage2CaloLayer2DERatio'),
                  input_dir_data_ = cms.untracked.string('L1T2016/L1TStage2CaloLayer2'),
                  input_dir_emul_ = cms.untracked.string('L1T2016EMU/L1TStage2CaloLayer2EMU')
)


