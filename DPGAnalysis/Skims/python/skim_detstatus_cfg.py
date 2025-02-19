import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("DPGAnalysis/Skims/DetStatus_cfi")


# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"

#adapt skimming
process.dcsstatus.DetectorType =  cms.vstring('TOB','BPIX','TECp','TECm')
process.dcsstatus.ApplyFilter =  cms.bool(True)
process.dcsstatus.DebugOn =  cms.untracked.bool(True)
process.dcsstatus.AndOr =  cms.bool(True)




process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/skim_detstatus_cfg.py,v $'),
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
# data
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/022/FC89901D-FFE6-DE11-AA38-001D09F25401.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/022/EED31843-06E7-DE11-B658-000423D6B5C4.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/022/56725260-03E7-DE11-8291-001D09F28EA3.root'
#MC
#'/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0000/026DF601-7916-DF11-8223-003048D46012.root',
#'/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0000/026DF601-7916-DF11-8223-003048D46012.root'
#RelVal
#'/store/relval/CMSSW_3_4_0_pre2/RelValMinBias/GEN-SIM-RECO/STARTUP3XY_V9-v1/0003/FE73FBC4-C0BD-DE11-9691-0026189438A5.root'
] )


process.source = cms.Source('PoolSource',
#    debugVerbosity = cms.untracked.uint32(0),
#    debugFlag = cms.untracked.bool(False),
    fileNames = myfilelist )


