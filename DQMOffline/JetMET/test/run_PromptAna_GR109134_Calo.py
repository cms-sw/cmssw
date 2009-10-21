#

import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# HCALNoise module
process.load("RecoMET.METProducers.hcalnoiseinfoproducer_cfi")
process.hcalnoise.refillRefVectors = cms.bool(True)
process.hcalnoise.hcalNoiseRBXCollName = "hcalnoise"
process.hcalnoise.requirePedestals = cms.bool(False)

# the task - JetMET objects
process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff")
process.jetMETAnalyzer.OutputMEsInRootFile = cms.bool(True)
process.jetMETAnalyzer.OutputFileName = cms.string('jetMETMonitoring_GR109134_Calo.root')
process.jetMETAnalyzer.DoJetPtAnalysis = cms.untracked.bool(True)
process.jetMETAnalyzer.caloMETAnalysis.allSelection       = cms.bool(True)
process.jetMETAnalyzer.caloMETNoHFAnalysis.allSelection   = cms.bool(True)
process.jetMETAnalyzer.caloMETHOAnalysis.allSelection     = cms.bool(True)
process.jetMETAnalyzer.caloMETNoHFHOAnalysis.allSelection = cms.bool(True)
process.jetMETAnalyzer.caloMETAnalysis.verbose            = cms.int32(0)

# the task - JetMET trigger
process.load("DQMOffline.Trigger.JetMETHLTOfflineSource_cfi")

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

# for igprof
#process.IgProfService = cms.Service("IgProfService",
#  reportFirstEvent            = cms.untracked.int32(0),
#  reportEventInterval         = cms.untracked.int32(25),
#  reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > igdqm.%I.gz")
#)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/EC0B693D-BE7C-DE11-8DF7-001D09F24763.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/E856A03C-BE7C-DE11-AB30-001D09F25460.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/E843B572-C77C-DE11-90FE-000423D6B444.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/E246E37E-C77C-DE11-9E60-001D09F28D54.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/C6F86257-C07C-DE11-AC8A-000423D99996.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/BA76158E-C97C-DE11-96D0-000423D986C4.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/B4B89A79-C27C-DE11-9D4B-000423D998BA.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/9A9C220F-E57C-DE11-9BA1-0019B9F72D71.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/987D120E-E57C-DE11-963C-001D09F27003.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/8E2C670F-E57C-DE11-8A57-001D09F253D4.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/78372356-C57C-DE11-A5C2-001D09F28D54.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/763AE7F1-BE7C-DE11-8316-0019B9F72BFF.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/748346F1-BE7C-DE11-A3E1-001D09F2A465.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/70776C19-E57C-DE11-BCA1-000423D9A212.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/6E4E38BF-C67C-DE11-BD9A-001D09F28F1B.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/4E6905E6-C37C-DE11-AA0E-0019B9F70468.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/4E31B083-BD7C-DE11-A224-001D09F2B2CF.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/408B3F3E-BE7C-DE11-841C-001D09F251CC.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/34FE333D-BE7C-DE11-A574-001D09F28F11.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/304AAC96-BD7C-DE11-84FA-0019B9F709A4.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/0C632371-C77C-DE11-B4A5-001D09F291D2.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/109/134/06FE3F28-C37C-DE11-AFC2-001D09F27067.root'
    )
)

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1000 )
)
process.Timing = cms.Service("Timing")

#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('jetMETAnalyzer'),
#    cout = cms.untracked.PSet(
#        default = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        jetMETAnalyzer = cms.untracked.PSet(
#            limit = cms.untracked.int32(100)
#        ),
#        noLineBreaks = cms.untracked.bool(True),
#        DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        #FwkJob = cms.untracked.PSet(
#        #    limit = cms.untracked.int32(0)
#        #),
#        threshold = cms.untracked.string('DEBUG')
#    ),
#    categories = cms.untracked.vstring('jetMETAnalyzer'),
#    destinations = cms.untracked.vstring('cout')
#)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    #outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('reco_DQM_GR109134_Calo.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False) ## default is false

)

process.p = cms.Path(process.hcalnoise
                     * process.jetMETHLTOfflineSource
                     * process.jetMETDQMOfflineSourceCosmic
                     * process.MEtoEDMConverter
                     * process.dqmStoreStats)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''

