import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

#~ from ApeEstimator.ApeEstimator.samples.GoodRunList_cff import LumisToProcess
#~ source.lumisToProcess = LumisToProcess

readFiles.extend( [
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/496/00000/2E9755C4-B72C-E511-B170-02163E01280D.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/559/00000/DCE6315D-A92C-E511-9A83-02163E013597.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/604/00000/28DF98A7-922A-E511-946F-02163E01354D.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/612/00000/BC61BA34-A52A-E511-9F83-02163E011AAF.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/638/00000/4C20E68C-0E2B-E511-9CD2-02163E0139A2.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/642/00000/1227560F-D22A-E511-BF3A-02163E0135B5.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/643/00000/52117043-D02C-E511-A494-02163E014543.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/721/00000/70D94B45-622C-E511-B9E7-02163E01366D.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/781/00000/E88CC2EF-9C2C-E511-BCD7-02163E0138EC.root',
		'/store/data/Run2015B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/251/883/00000/543B29E1-B02D-E511-9137-02163E012916.root',
		] );



secFiles.extend( [
               ] )

