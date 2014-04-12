#

import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# the task
process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff")
process.jetMETAnalyzer.OutputMEsInRootFile = cms.bool(True)
process.jetMETAnalyzer.OutputFileName = cms.string('jetMETMonitoring_cruzet100945.root')
process.jetMETAnalyzer.DoJetPtAnalysis = cms.untracked.bool(True)
process.jetMETAnalyzer.caloMETAnalysis.allSelection       = cms.bool(True)
process.jetMETAnalyzer.caloMETNoHFAnalysis.allSelection   = cms.bool(True)
process.jetMETAnalyzer.caloMETHOAnalysis.allSelection     = cms.bool(True)
process.jetMETAnalyzer.caloMETNoHFHOAnalysis.allSelection = cms.bool(True)

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
    '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/FA72B935-0960-DE11-A902-000423D98DB4.root'
    #'/store/data/Commissioning09/Calo/RECO/v3/000/100/945/0C547BAF-0C60-DE11-83C3-000423D98868.root'
    #'/store/data/CRUZET09/Calo/RECO/v1/000/098/154/EADF3BE3-BE4F-DE11-8BB8-000423D9870C.root'
    #'file:/tmp/hatake/EADF3BE3-BE4F-DE11-8BB8-000423D9870C.root'
    )
)
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/FA72B935-0960-DE11-A902-000423D98DB4.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/EA597588-0F60-DE11-938A-001D09F251B8.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/B80B1DA7-0560-DE11-8850-000423D6CA02.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/A2A21790-0A60-DE11-8231-001617E30D40.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/94D38392-0A60-DE11-8DE9-0019DB2F3F9A.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/8C1F34B5-0C60-DE11-9413-000423D985B0.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/6243D9DC-0960-DE11-8104-001617C3B5F4.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/52F63D01-2460-DE11-8DB4-001D09F24489.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/4EDF8E52-0B60-DE11-A6CB-000423D98930.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/301E3B65-0D60-DE11-B51C-000423D94908.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/22E1ECD9-0E60-DE11-B9EB-001D09F28D54.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/22BEF6AE-0C60-DE11-B76D-001617C3B6C6.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/1EA1811B-0E60-DE11-A6F0-001617C3B76E.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/1CE0FB64-0D60-DE11-9F26-000423D98BE8.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/1AA3E288-0F60-DE11-8900-001D09F24399.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/12CCE9D9-0E60-DE11-95FD-001D09F2A690.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/10075189-0F60-DE11-B1E5-001D09F2932B.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/0C547BAF-0C60-DE11-83C3-000423D98868.root',
#        '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/06B387FF-0B60-DE11-94A9-000423D95220.root'

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1000 )
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
    fileName = cms.untracked.string('reco_DQM_cruzet100945.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)

#process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
#process.load("DQMOffline.JetMET.jetMETDQMStoreClean_cff");

process.p = cms.Path(process.jetMETDQMOfflineSourceCosmic
                     * process.MEtoEDMConverter
                     * process.dqmStoreStats)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''

