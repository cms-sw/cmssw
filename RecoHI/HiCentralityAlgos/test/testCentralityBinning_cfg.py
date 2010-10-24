process = cms.Process('ANALYSIS')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("dcache:/pnfs/cmsaf.mit.edu/t2bat/cms/store/himc/Fall10/AMPT_Default_MinBias_2760GeV/GEN-SIM-RECO/MC_38Y_V12-v1/0002/F8E134B2-77D6-DF11-BFE1-001B243DE10F.root")
                            )


process.load('FrontierConditions_GlobalTag_Temp_cff')
process.GlobalTag.globaltag = 'MC_38Y_V8::All'

process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("PixelHits"),
    nonDefaultGlauberModel = cms.string("AMPT_2760GeV"),
    centralitySrc = cms.InputTag("hiCentrality")
    )


process.ana = cms.EDAnalyzer('AnalyzerWithCentrality')

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('plots2.root')
                                   )

process.p = cms.Path(process.ana)

