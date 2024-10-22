import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [
       "/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/826/00000/E23F3B37-8C8B-E811-9E54-FA163EAD4CB1.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/827/00000/884C3F8F-8C8B-E811-8B4F-02163E019F19.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/829/00000/488AF044-8C8B-E811-B07B-FA163EF05F9D.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/830/00000/7E0F6EC0-918B-E811-85ED-02163E019F09.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/831/00000/A0614791-968B-E811-A59D-FA163E7750CB.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/833/00000/F0580EF1-968B-E811-8FB6-FA163E9C118B.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/835/00000/B0823EE4-9D8B-E811-A96B-FA163E7750CB.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/838/00000/04D2AA3F-A68B-E811-84B1-02163E0152A4.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/840/00000/06074A5B-E18B-E811-A5A5-FA163E53A37E.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/840/00000/1252EF18-098C-E811-9258-FA163EBF76CA.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/840/00000/5881721E-E38B-E811-AF87-02163E00BA55.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/840/00000/78CBC170-EB8B-E811-BEC2-FA163E23196D.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/840/00000/7C74081A-EA8B-E811-A6F8-02163E017F89.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/840/00000/BE7978CE-EC8B-E811-97AD-FA163EF5922E.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/841/00000/080B1B2D-ED8B-E811-99FA-FA163EB165D3.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/841/00000/9CD23BA8-F58B-E811-8F50-FA163E2C6343.root", 
       #"/store/data/Run2018C/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v3/000/319/841/00000/F417EAAE-FD8B-E811-8132-FA163EA194D3.root", 
       ] );



secFiles.extend( [
               ] )

