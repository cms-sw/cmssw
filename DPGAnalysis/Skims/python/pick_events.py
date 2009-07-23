import FWCore.ParameterSet.Config as cms

process = cms.Process("PICKEVENT")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/063B4FB4-1D82-DD11-97C1-000423D996B4.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/0C70D3E2-1B82-DD11-B005-000423D6CA42.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/16C53F54-1B82-DD11-A2B8-000423DD2F34.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/1AAAC808-9082-DD11-880E-001617E30D06.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/245F8A2A-1B82-DD11-B852-000423D8F63C.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/48AB7D04-1A82-DD11-B311-000423D6AF24.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/5C821132-1D82-DD11-9C65-000423D94990.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/5E91B795-2082-DD11-B973-001617DC1F70.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/6A85979B-1C82-DD11-B441-001617E30E28.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/7879CAA6-1882-DD11-82B4-000423D98750.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/7C5A3EA0-1B82-DD11-87C6-000423D9997E.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/A645BD82-1882-DD11-B023-000423D98EC8.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/A874067D-8F82-DD11-87AB-001617C3B710.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/AA880EB4-8F82-DD11-BD16-001617C3B778.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/B60CBD20-1C82-DD11-84E2-00161757BF42.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/B8B25E53-9082-DD11-BF63-001617E30D40.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/DC748397-1982-DD11-8AAC-000423D94494.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/EE4BD502-9182-DD11-9BE2-000423D98800.root',
       '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/571/F8144871-1982-DD11-BB2F-000423D99E46.root'
      )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Configuration.EventContent.EventContentCosmics_cff")
process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW')),
    fileName = cms.untracked.string("/tmp/schmittm/cscunpacker_crash.root"),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('mySkim')
       )
)
process.FEVT.outputCommands.append('keep FEDRawDataCollection_*_*_*')




#--------------------------------------------------
#   Pick a range of events
#     (includes the first and last ones specified)
#--------------------------------------------------
process.pickEvents = cms.EDFilter(
    "PickEvents",
    whichRun = cms.untracked.int32(62571),
    whichEventFirst = cms.untracked.int32(150250),
    whichEventLast  = cms.untracked.int32(150280)
    )




process.mySkim  = cms.Path( process.pickEvents )

process.outpath = cms.EndPath(process.FEVT)
