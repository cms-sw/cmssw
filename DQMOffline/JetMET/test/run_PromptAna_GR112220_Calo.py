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
process.jetMETAnalyzer.OutputFileName = cms.string('jetMETMonitoring_GR112220_Calo.root')
process.jetMETAnalyzer.DoJetPtAnalysis = cms.untracked.bool(True)
process.jetMETAnalyzer.caloMETAnalysis.allSelection       = cms.bool(True)
process.jetMETAnalyzer.caloMETNoHFAnalysis.allSelection   = cms.bool(True)
process.jetMETAnalyzer.caloMETHOAnalysis.allSelection     = cms.bool(True)
process.jetMETAnalyzer.caloMETNoHFHOAnalysis.allSelection = cms.bool(True)
process.jetMETAnalyzer.caloMETAnalysis.verbose     = cms.int32(0)

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
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/F0B768A4-5E93-DE11-B222-000423D94524.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/E25D4F9E-5E93-DE11-8D3E-003048D2C0F0.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/E0132A9C-5E93-DE11-ACA0-0019DB29C614.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/CAC87C9B-5E93-DE11-85AF-000423D98DC4.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/BC0F3371-5893-DE11-85B4-001D09F241B4.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/9866B39E-5E93-DE11-AE6C-003048D37456.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/70B5C597-5E93-DE11-BC9C-000423D9863C.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/6E1DF6A1-5E93-DE11-BD4A-000423D987FC.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/5A82B29E-5E93-DE11-8381-003048D37560.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/2EEFED9A-5E93-DE11-A6F7-000423D98834.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/1C85A466-5893-DE11-8FB5-001D09F252F3.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/142D5D99-5E93-DE11-961E-000423D94C68.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/0AABE3B2-5993-DE11-9EE2-000423D98EA8.root',
    '/store/data/CRAFT09/Calo/RECO/v1/000/112/220/04C0659F-5E93-DE11-88AC-000423D6B444.root'
    )
)

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32( 100 )
    input = cms.untracked.int32( 100000 )
)
process.Timing = cms.Service("Timing")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('jetMETAnalyzer'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        jetMETAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(100)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        #FwkJob = cms.untracked.PSet(
        #    limit = cms.untracked.int32(0)
        #),
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('jetMETAnalyzer'),
    destinations = cms.untracked.vstring('cout')
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    #outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('reco_DQM_GR112220_Calo.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)

process.p = cms.Path(process.hcalnoise
                     * process.jetMETHLTOfflineSource
                     * process.jetMETDQMOfflineSourceCosmic
                     * process.MEtoEDMConverter
                     * process.dqmStoreStats)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''

