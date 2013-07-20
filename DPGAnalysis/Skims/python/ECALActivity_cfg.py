import FWCore.ParameterSet.Config as cms

process = cms.Process("RESKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# activate the following lines to get some output
#process.MessageLogger.destinations = cms.untracked.vstring("cout")
#process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO"))
process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)


process.skimming = cms.EDFilter("ECALActivity",
    ebrechitcollection =   cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    eerechitcollection =   cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EBnum = cms.untracked.int32(7),
    EBthresh = cms.untracked.double(0.3),
    EEnum = cms.untracked.int32(5), 
    EEthresh = cms.untracked.double(0.5),
    ETOTnum = cms.untracked.int32(3),
    ETOTthresh = cms.untracked.double(0.8),
    applyfilter = cms.untracked.bool(True)                            
)                    

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(0),
    debugFlag = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(
'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/bit40or41skim.root'
#'/store/data/Commissioning08/BeamHalo/RECO/StuffAlmostToP5_v1/000/061/642/10A0FE34-A67D-DD11-AD05-000423D94E1C.root'#
#
#'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FED0EFCD-AB87-DE11-9B72-000423D99658.root',
#'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FC629BD2-CF87-DE11-9077-001D09F25438.root',
#'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FC38EE75-BD87-DE11-822A-001D09F253C0.root',
#'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FC1CB101-A487-DE11-9F10-000423D99660.root'
#
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/D266D139-D871-DE11-A709-001D09F28F0C.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/CA27788D-E871-DE11-8B46-001D09F276CF.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/AC5633B2-D471-DE11-9B3A-001D09F252F3.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/9CD957E7-D071-DE11-B6AE-001D09F252F3.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/94BF68F7-D171-DE11-902B-000423D986A8.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/7838FE1E-C771-DE11-9FD5-000423D98950.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/56632803-DD71-DE11-BAF5-000423D9870C.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/42A67CB9-E971-DE11-AA86-001D09F252F3.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/407225D3-D071-DE11-809B-001D09F297EF.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/3E5E1CF0-D271-DE11-AC2B-000423D94700.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/2C57E916-D071-DE11-AF0E-001D09F24E39.root',
#'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/228896A5-E571-DE11-A60B-001D09F2AF96.root'
))

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/ECALActivity_cfg.py,v $'),
    annotation = cms.untracked.string('ECALActivity skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ECALActivity.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RECO'),
    	      filterName = cms.untracked.string('ECALActivity')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

process.p = cms.Path(process.skimming)
process.e = cms.EndPath(process.out)

