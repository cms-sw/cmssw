from RecoLuminosity.LumiProducer.expressLumiProducer_cff import *
process = cms.Process("dbprodtest")
process.load('RecoLuminosity/LumiProducer/expressLumiProducer_cff')
process.source= cms.Source("PoolSource",
             processingMode=cms.untracked.string('RunsAndLumis'),        
             fileNames=cms.untracked.vstring('file:/data/cmsdata/009F3522-D604-E111-A08D-003048F1183E.root'),            
)
process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testExpressLumiProd.root')
)
process.p1 = cms.Path(process.expressLumiProducer)
process.e = cms.EndPath(process.out)
