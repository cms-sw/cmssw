import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.skimming = cms.EDFilter("PhysDecl",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(True),
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")

)


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/skim_physdecl_cfg.py,v $'),
    annotation = cms.untracked.string('PhysDecl skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('mytest_physdecl.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RECO'),
    	      filterName = cms.untracked.string('PhysDecl')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

process.p = cms.Path(process.skimming)
process.e = cms.EndPath(process.out)

myfilelist = cms.untracked.vstring()

#myfilelist.extend( ['file:evenmorefile1.root','file:evenmorefile2.root'] )
#process.source = cms.Source("PoolSource",
#    debugVerbosity = cms.untracked.uint32(0),
#    debugFlag = cms.untracked.bool(False),
#    fileNames = myfilelist
#))
myfilelist.extend( [
'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/CCD51E51-52D4-DE11-A51D-001617C3B654.root',
'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/C2C8880A-55D4-DE11-8883-001617C3B6DE.root',
'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/B047990A-55D4-DE11-A6A0-001D09F23944.root',
'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/42C5EC0B-55D4-DE11-ABA3-001617DBD556.root'
#'/store/data/BeamCommissioning09/Cosmics/RECO/v2/000/121/550/AAC5F95D-5BD4-DE11-BA3F-000423D94524.root'
#'/store/data/BeamCommissioning09/RandomTriggers/RAW/v1/000/121/550/845D19D0-57D4-DE11-8FF8-001D09F23944.root'
] )
process.source = cms.Source('PoolSource',
    debugVerbosity = cms.untracked.uint32(0),
    debugFlag = cms.untracked.bool(False),
    fileNames = myfilelist )

