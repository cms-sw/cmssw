import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("DPGAnalysis/Skims/skim_detstatus_cfi")


# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"

#adapt skimming
process.dcsstatus.DetectorType =  cms.vstring('TOB','BPIX','TECp','TECm')
process.dcsstatus.ApplyFilter =  cms.bool(True)
process.dcsstatus.DebugOn =  cms.untracked.bool(True)




process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/skim_detstatus_cfg.py,v $'),
    annotation = cms.untracked.string('DCSStatus skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('mytest_dcsstatus.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RECO'),
    	      filterName = cms.untracked.string('DCSStatus')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

process.p = cms.Path(process.dcsstatus)
process.e = cms.EndPath(process.out)

myfilelist = cms.untracked.vstring()

#myfilelist.extend( ['file:evenmorefile1.root','file:evenmorefile2.root'] )
#process.source = cms.Source("PoolSource",
#    debugVerbosity = cms.untracked.uint32(0),
#    debugFlag = cms.untracked.bool(False),
#    fileNames = myfilelist
#))
myfilelist.extend( [
'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/022/FC89901D-FFE6-DE11-AA38-001D09F25401.root',
'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/022/EED31843-06E7-DE11-B658-000423D6B5C4.root',
'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/022/56725260-03E7-DE11-8291-001D09F28EA3.root'
#'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/CCD51E51-52D4-DE11-A51D-001617C3B654.root',
#'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/C2C8880A-55D4-DE11-8883-001617C3B6DE.root',
#'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/B047990A-55D4-DE11-A6A0-001D09F23944.root',
#'/store/express/BeamCommissioning09/OfflineMonitor/FEVTHLTALL/v1/000/121/550/42C5EC0B-55D4-DE11-ABA3-001617DBD556.root'
#'/store/data/BeamCommissioning09/Cosmics/RECO/v2/000/121/550/AAC5F95D-5BD4-DE11-BA3F-000423D94524.root'
#'/store/data/BeamCommissioning09/RandomTriggers/RAW/v1/000/121/550/845D19D0-57D4-DE11-8FF8-001D09F23944.root'
] )
process.source = cms.Source('PoolSource',
    debugVerbosity = cms.untracked.uint32(0),
    debugFlag = cms.untracked.bool(False),
    fileNames = myfilelist )

