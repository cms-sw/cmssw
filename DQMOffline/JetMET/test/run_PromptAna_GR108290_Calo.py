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
#process.jetMETAnalyzer.OutputFileName = cms.string('jetMETMonitoring_GR108290_Calo.root')
process.jetMETAnalyzer.OutputFileName = cms.string('jetMETMonitoring_TTbar.root')
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
#    'file:/home/xv/sordini/QCD_Pt170_summer09_31X_V3-v1_GEN-SIM-RECO.root'
    '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0008/2C8CD8FE-B6B5-DE11-ACB8-001D09F2905B.root'
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/FC70C563-1678-DE11-92EC-000423D999CA.root'
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/F8AF03F6-1978-DE11-829B-001D09F252F3.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/F83D885F-1678-DE11-BAC1-000423D951D4.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/F26AA2D6-1278-DE11-B9C5-000423D98FBC.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/E6B19C9E-1A78-DE11-B0DC-000423D98EC8.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/E417AC6B-1678-DE11-91E2-000423D986A8.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/E0C2C65E-1678-DE11-B755-000423D98BC4.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/DE34EE8D-1878-DE11-A20C-001D09F24024.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/C0115BD1-1778-DE11-9D63-001D09F24DDA.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/B61073CE-1778-DE11-BAC0-001D09F244BB.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/B4A66260-1678-DE11-AF5A-000423D944F8.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/AC18D41F-1778-DE11-898F-000423D94494.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/A8842AFA-1478-DE11-9C71-000423D991D4.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/9EC2FB1F-1778-DE11-90D3-001D09F34488.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/948CEE47-1478-DE11-8610-000423D951D4.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/9243415C-1B78-DE11-8650-001D09F24DDA.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/808C5C48-1478-DE11-BE29-000423D944F0.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/78FC8993-1F78-DE11-B79F-000423D99160.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/76A40786-1878-DE11-A8BB-001D09F23E53.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/76A299CE-1778-DE11-8CBE-000423D98E54.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/709AB95E-1678-DE11-AF06-000423D98E6C.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/689F8D60-1678-DE11-B981-000423D99896.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/68457387-1878-DE11-BD30-001D09F2503C.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/64E9C35C-1B78-DE11-ADC6-001D09F24D8A.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/60E22226-1778-DE11-A958-001D09F2983F.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/5CF42529-1778-DE11-B52D-0019B9F72D71.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/5C7C76AF-2178-DE11-B49A-000423D998BA.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/586703D2-1778-DE11-8DDC-001D09F24399.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/52776364-1678-DE11-B9EA-000423D990CC.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/4C4D1F60-1678-DE11-A9BD-000423D99658.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/4A5576CF-1778-DE11-AF6A-001D09F25438.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/40CF75CE-1778-DE11-95E1-001D09F24DA8.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/3E56DB20-1778-DE11-9041-001D09F2B2CF.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/363856D0-1778-DE11-8E60-000423D991F0.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/2AF0D062-2278-DE11-A9EF-001D09F2437B.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/2AD949CF-1778-DE11-AB52-001D09F29114.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/1C584562-2278-DE11-9835-001D09F2538E.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/1C55F321-1778-DE11-A1CF-001D09F24EAC.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/0E943F48-1478-DE11-849B-000423D98E6C.root',
#    '/store/data/Commissioning09/Calo/RECO/v6/000/108/290/0AFFD0CE-1778-DE11-87A6-001D09F2AF96.root',
#    '/store/data/Commissioning09/Calo/RECO/v3/000/100/945/FA72B935-0960-DE11-A902-000423D98DB4.root'
    )
)

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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
    fileName = cms.untracked.string('reco_DQM_GR108290_Calo.root')
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

