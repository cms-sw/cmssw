from RecoLuminosity.LumiProducer.lumiProducer_cff import *
process = cms.Process("dbprodtest")
process.load('RecoLuminosity/LumiProducer/lumiProducer_cff')
process.source= cms.Source("PoolSource",
             processingMode=cms.untracked.string('RunsAndLumis'),        
             fileNames=cms.untracked.vstring(
'/store/data/Run2010B/Commissioning/RECO/PromptReco-v2/000/149/181/FC28B5D3-13E4-DF11-B4F4-0030486780E6.root'
#,'/store/data/Run2010B/Commissioning/RECO/PromptReco-v2/000/149/181/ECD8B91F-D8E3-DF11-9C27-0019B9F70607.root',
#'/store/data/Run2010B/Commissioning/RECO/PromptReco-v2/000/149/181/C87FB7A7-D4E3-DF11-AD73-001D09F251BD.root',
#'/store/data/Run2010B/Commissioning/RECO/PromptReco-v2/000/149/181/C280CBD5-D8E3-DF11-8B33-001D09F24D8A.root'
),            
)
process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testLumiProd-149181.root')
)
process.p1 = cms.Path(process.lumiProducer)
process.e = cms.EndPath(process.out)
