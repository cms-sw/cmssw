import FWCore.ParameterSet.Config as cms

process = cms.Process("RaddamMuon")

process.load("Calibration.HcalAlCaRecoProducers.alcahbhemuon_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'root://xrootd.unl.edu//store/mc/Phys14DR/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RECO/PU20bx25_tsg_castor_PHYS14_25_V1-v1/10000/184C1AC9-A775-E411-9196-002590200824.root'
        )

                            )

process.muonOutput = cms.OutputModule("PoolOutputModule",
                                      fileName = cms.untracked.string('PoolOutput.root'),
                                      outputCommands = cms.untracked.vstring('keep *_HBHEMuonProd_*_*')
                                      )

process.p = cms.Path(process.HBHEMuonProd)
process.e = cms.EndPath(process.muonOutput)
