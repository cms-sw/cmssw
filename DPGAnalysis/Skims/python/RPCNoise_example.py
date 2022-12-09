import FWCore.ParameterSet.Config as cms

process = cms.Process("USER")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'auto:run3_data_prompt'
process.prefer("GlobalTag")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/00718365-02A6-DD11-86BC-000423D98E54.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/00725EB9-22A6-DD11-8EC1-001617DC1F70.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/00A93B38-26A6-DD11-8676-000423D98F98.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/00EB701D-24A6-DD11-9AA1-001617E30D38.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/00FC7A6B-E3A5-DD11-A4D1-001617DF785A.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/0239F671-DCA5-DD11-9268-000423D98844.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/02450760-F4A5-DD11-B709-000423D6BA18.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/0260B87A-49A6-DD11-9731-000423D992A4.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/02914A0F-29A6-DD11-BD17-000423D985B0.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/02C0EAC4-62A6-DD11-868F-000423D6CA02.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/02CD99AF-1BA6-DD11-B71E-000423D992DC.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/02F1E96F-50A6-DD11-938D-0019DB29C614.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/02FA5CED-56A6-DD11-8F63-001617E30D52.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/044D708C-5EA6-DD11-BFA5-0030487D0D3A.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/0457B4E8-13A6-DD11-816E-000423D98920.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/046760F2-0CA6-DD11-B377-000423D94A20.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/04766D63-5CA6-DD11-B07D-000423D98834.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/047C6309-29A6-DD11-AC8E-000423D6B42C.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/04861900-22A6-DD11-90DA-000423D944F8.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/04B99356-5AA6-DD11-950F-0030487A3C9A.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/06192894-3FA6-DD11-9A2B-000423D990CC.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/062D36C0-3CA6-DD11-BB50-001617C3B69C.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/066D62D5-56A6-DD11-970B-001617C3B6CC.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/0681071B-5DA6-DD11-A9A1-000423D99660.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/0684A267-09A6-DD11-B4FA-000423D98B5C.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/06AB249F-46A6-DD11-96C1-000423D99CEE.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/06BFD55E-1CA6-DD11-AC5A-001D09F2915A.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/08003087-DEA5-DD11-A4EE-000423D98B5C.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/021/0866B1DE-FEA5-DD11-8F9F-0030487C6062.root'
      )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)

#### output module
process.load("Configuration.EventContent.EventContentCosmics_cff")
process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW')),
    fileName = cms.untracked.string("RPCNoise_test.root"),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('noiseEvents')
       )
)
process.FEVT.outputCommands.append('keep *_*_*_*')
process.FEVT.outputCommands.append('keep FEDRawDataCollection_*_*_*')
process.FEVT.outputCommands.append('keep *_muonCSCDigis_*_*')
process.FEVT.outputCommands.append('keep *_muonDTDigis_*_*')
process.FEVT.outputCommands.append('keep *_muonRPCDigis_*_*')
process.FEVT.outputCommands.append('keep *_rpcRecHits_*_*')


#============================================================
# the filter
#============================================================
process.check = cms.EDFilter(
    "RPCNoise",
    fillHistograms = cms.untracked.bool(True),
    histogramFileName = cms.untracked.string('histos_test.root'),
    nRPCHitsCut  = cms.untracked.int32(40),
    nCSCWiresCut  = cms.untracked.int32(10),
    nCSCStripsCut  = cms.untracked.int32(50),
    nDTDigisCut  = cms.untracked.int32(40)
)

#process.noiseEvents = cms.Path(process.muonCSCDigis *
#                               process.muonRPCDigis *
#                               process.rpcRecHits *
#                               process.check)


process.muondigis = cms.Sequence(process.csctfDigis+process.dttfDigis+process.gctDigis+process.gtDigis+
                                 process.gtEvmDigis+
                                 process.muonCSCDigis+process.muonDTDigis+process.muonRPCDigis)

process.noiseEvents = cms.Path(process.muondigis *
                               process.muonlocalreco *
                               process.check)

process.outpath = cms.EndPath(process.FEVT)
