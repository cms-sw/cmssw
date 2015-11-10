import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

#~ from ApeEstimator.ApeEstimator.samples.GoodRunList_cff import LumisToProcess
#~ source.lumisToProcess = LumisToProcess

readFiles.extend( [
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/496/00000/14E79ACC-B12C-E511-A988-02163E01387D.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/496/00000/968A90CF-B12C-E511-A3DF-02163E0125D6.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/559/00000/62FCBC3C-A82C-E511-9AEE-02163E011D30.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/604/00000/A4A23DAE-982A-E511-A8D0-02163E0133A7.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/612/00000/1ADBBF67-A92A-E511-A307-02163E0136E2.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/638/00000/4E24A7C3-062B-E511-8BBA-02163E012B30.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/642/00000/B64059B9-DF2A-E511-B6A3-02163E0146EB.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/643/00000/1ECE758E-BF2C-E511-9E9F-02163E0123C0.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/643/00000/343AFCAD-BF2C-E511-9BE3-02163E013553.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/643/00000/6254A38C-BF2C-E511-ACDE-02163E011D37.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/721/00000/E6DF7C94-562C-E511-AF51-02163E013436.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/781/00000/D47685E4-AA2C-E511-9C9D-02163E011CD6.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/883/00000/18110DE3-4A2D-E511-BF57-02163E0128BF.root',
       '/store/data/Run2015B/SingleMuon/ALCARECO/TkAlMuonIsolated-PromptReco-v1/000/251/883/00000/AEA22FF8-4A2D-E511-A82E-02163E0133E3.root', 
       ] );



secFiles.extend( [
               ] )

