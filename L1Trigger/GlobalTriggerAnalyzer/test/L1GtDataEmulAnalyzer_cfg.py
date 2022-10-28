#
# cfg file for: 
#
#   Run the L1 GT emulator on the unpacked GCT and GMT data.
#   Compare the GT data records with the GT emulated records


import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("RunL1GtDataEmulAnalyzer")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_DataEmulAnalyzer_source.root')
)

process.PoolSource.fileNames = ['/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/04AA7F35-C426-DD11-B047-001D09F2516D.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/04F935C6-C426-DD11-AE42-001D09F24637.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/0663785C-C626-DD11-8118-000423D94534.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/068D1384-C526-DD11-A027-000423D94A04.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/069F542D-C426-DD11-B4FD-001D09F290CE.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/088C9FC0-C426-DD11-80A5-000423D99CEE.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/0A89B10E-C526-DD11-886F-001D09F2532F.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/0E2F9E85-C626-DD11-B460-001D09F231B0.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/1012C6C0-C426-DD11-8E75-000423D6B444.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/1212CF82-C526-DD11-8840-001617E30D0A.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/14287A88-C526-DD11-978F-000423D99896.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/16AE8487-C526-DD11-88F5-001D09F24498.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/1A728D7F-C526-DD11-9363-001617C3B6E2.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/1C63F034-C426-DD11-98A0-001D09F24F65.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/1E4EC288-C526-DD11-94B0-001D09F2525D.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/20C9B509-C526-DD11-BD8C-001D09F244DE.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/2642ADCA-C426-DD11-A971-001D09F2906A.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/2A9FB786-C526-DD11-BB6D-001617C3B6CC.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/2C1C5209-C526-DD11-A4B5-001D09F24D67.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/2E4FB3C8-C426-DD11-91E0-0019B9F705A3.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/2E53B076-C526-DD11-AF83-001D09F2A465.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/325C9484-C626-DD11-9123-001D09F24F65.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/34D18586-C526-DD11-AE6E-001617C3B69C.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/3A4BB185-C526-DD11-A030-0019B9F705A3.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/3AFE8409-C526-DD11-9F2C-001617DBD332.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/3CDBC0B0-C626-DD11-9A8F-001D09F2525D.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/3E44DA7D-C526-DD11-8555-000423D951D4.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/4058E48A-C526-DD11-97FF-001617E30F56.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/44F9F281-C526-DD11-B6CE-000423D99AA2.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/48D9347F-C526-DD11-B90E-001617E30D52.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/4EF5B58B-C526-DD11-B608-001D09F2A49C.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/50E8D437-C426-DD11-A66B-001D09F251FE.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/5473F70C-C526-DD11-84D8-001D09F24DDA.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/5690A985-C626-DD11-8F78-001D09F2932B.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/58B44484-C526-DD11-BE92-000423D9939C.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/5C95A980-C526-DD11-AA27-001617DBD332.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/5CAA59AB-C626-DD11-9451-001D09F25456.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/5EA14E0A-C526-DD11-AF5C-001D09F28E80.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/5EADB005-C526-DD11-B7A1-001D09F24399.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/668DEA01-C526-DD11-B7DB-001D09F2546F.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/66F05B87-C526-DD11-ACAA-000423D94E70.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/6ABEE90D-C626-DD11-B9E9-0030487A18F2.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/6C142B89-C526-DD11-BC79-001617C3B706.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/6C36B389-C526-DD11-A4FF-001D09F28F25.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/6C6E8381-C526-DD11-949A-000423D98B28.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/6E15E00D-C526-DD11-8DBB-001D09F24EC0.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/6E28C8BD-C426-DD11-8219-000423D944F8.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/70E8F2C1-C426-DD11-9B3E-001D09F2503C.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/72CEF283-C526-DD11-B393-000423D98DC4.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/76D1AB05-C526-DD11-A4DB-001D09F2841C.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/7AAA5B7D-C526-DD11-9E9C-000423D944FC.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/7C1F29C1-C426-DD11-81E4-000423D944FC.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/7CDE918E-C526-DD11-8B51-0019B9F707D8.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/8037767E-C626-DD11-839B-001D09F2960F.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/8060D709-C526-DD11-B39E-001D09F251E0.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/82065F1A-C626-DD11-A0AC-001D09F231C9.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/84D88E85-C526-DD11-A46F-001617C3B6CE.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/8620AD09-C526-DD11-B0DC-001D09F28F11.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/863AE5CB-C426-DD11-8154-001D09F28E80.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/884FA985-C626-DD11-82A9-001D09F291D7.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/9692F0C6-C426-DD11-AD86-0019B9F72BFF.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/98FBFF87-C526-DD11-A0AE-001D09F24EC0.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/A8AB9B85-C626-DD11-BCDE-001D09F2305C.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/AA0E6689-C526-DD11-9FD8-001D09F24DA8.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/AE1B864B-C326-DD11-AF5A-000423D986C4.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/AE437109-C526-DD11-BC07-001D09F2438A.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/AEF6E086-C526-DD11-8326-0019B9F704D1.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/B4710DA7-C526-DD11-B659-001D09F23A6B.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/B4EF9581-C626-DD11-8E4A-001D09F28E80.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/BAA4138B-C526-DD11-870F-000423D944F0.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/BAB50884-C626-DD11-8418-001D09F2525D.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/C4147609-C526-DD11-A649-001D09F292D1.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/C6B5B17F-C526-DD11-8145-000423D99660.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/CA88975E-C526-DD11-8ED6-001617C3B76E.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/CEB69E7F-C526-DD11-9E94-000423D98BC4.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/CECA5B7C-C526-DD11-A158-001617DBD5AC.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/D0352C1C-C526-DD11-9B7B-001D09F29538.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/D2D36EC1-C426-DD11-B413-001D09F250AF.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/D42DF0D3-C526-DD11-9A01-001D09F2A49C.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/D6CAAA9D-C526-DD11-9F52-000423D98750.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/DC14738A-C526-DD11-8814-000423D99264.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/DCE0C480-C526-DD11-993D-000423D9997E.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/DEE7C481-C526-DD11-8E06-001617E30D12.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/E435D97C-C626-DD11-969E-001D09F251E0.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/E43DDA88-C526-DD11-B226-001D09F2AD4D.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/E839788B-C526-DD11-9487-000423D99BF2.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/EAA00680-C526-DD11-AEA9-000423D98800.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/F23E5C02-C526-DD11-81FC-001D09F23D1D.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/F2BD4C0B-C526-DD11-A42F-001D09F23A3E.root', 
    '/store/data/2008/5/20/T0ReReco-GlobalCruzet1-A-v1/0004/FA19908B-C526-DD11-A839-0019B9F730D2.root']


# load and configure modules

process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("L1Trigger.Configuration.L1Config_cff")

# L1 menu    
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu_CRUZET200805_gr7_muon_cff")

# Global Trigger emulator

import L1Trigger.GlobalTrigger.gtDigis_cfi
process.l1GtEmulDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()

# block GCT input and the technical triggers (only FDL and GMT active) 0x0101
process.l1GtParameters.DaqActiveBoards = 0x010d

# block GMT input (0xdd12)
#process.l1GtParameters.DaqActiveBoards = 0x00FF
        
# block both GCT and GMT (FDL and techTrig active)
#process.l1GtParameters.DaqActiveBoards = 0x0003

# input tag for GMT readout collection: 
process.l1GtEmulDigis.GmtInputTag = 'gtDigis'

# input tag for GCT readout collections: 
#process.l1GtEmulDigis.GctInputTag = 'gctDigis'

# logical flag to produce the L1 GT DAQ readout record
#     if true, produce the record (default)
#process.l1GtEmulDigis.ProduceL1GtDaqRecord = False
    
# logical flag to produce the L1 GT EVM readout record
#     if true, produce the record (default)
#process.l1GtEmulDigis.ProduceL1GtEvmRecord = False

# logical flag to produce the L1 GT object map record
#     if true, produce the record (default)
#process.l1GtEmulDigis.ProduceL1GtObjectMapRecord = False

# logical flag to write the PSB content in the  L1 GT DAQ record
#     if true, write the PSB content in the record (default)
#process.l1GtEmulDigis.WritePsbL1GtDaqRecord = False

# logical flag to read the technical trigger records
#     if true, it will read via getMany the available records (default)
#process.l1GtEmulDigis.ReadTechnicalTriggerRecords = False

# number of "bunch crossing in the event" (BxInEvent) to be emulated
# symmetric around L1Accept (BxInEvent = 0):
#    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
# even numbers (except 0) "rounded" to the nearest lower odd number
# negative value: emulate TotalBxInEvent as given in EventSetup  
#process.l1GtEmulDigis.EmulateBxInEvent = 3


# Global Trigger report

import L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi
process.l1GtTrigReportData = L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi.l1GtTrigReport.clone()

process.l1GtTrigReportData.L1GtRecordInputTag = 'gtDigis'

#
import L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi
process.l1GtTrigReportEmul = L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi.l1GtTrigReport.clone()

process.l1GtTrigReportEmul.L1GtRecordInputTag = 'l1GtEmulDigis'

#
# compare the L1 GT data and emulator digis
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtDataEmulAnalyzer_cfi")
process.l1GtDataEmulAnalyzer.L1GtEmulInputTag = 'l1GtEmulDigis'

# paths to be run
process.p = cms.Path(process.l1GtEmulDigis*process.l1GtDataEmulAnalyzer*process.l1GtTrigReportData*process.l1GtTrigReportEmul)

# services

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string('INFO'),
    INFO = cms.untracked.PSet(
        #limit = cms.untracked.int32(-1)
        limit = cms.untracked.int32(1000)
    )#,
    
    #threshold = cms.untracked.string('DEBUG'), ## DEBUG 

    #DEBUG = cms.untracked.PSet( ## DEBUG, all messages  
    #
    #    limit = cms.untracked.int32(-1)
    #)
)
process.MessageLogger.debugModules = ['l1GtEmulDigis']

# histogram service
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1GtDataEmulAnalyzer.root')
)

# summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# output 
process.outputL1GtDataEmul = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_DataEmulAnalyzer_output.root'),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtDataDigis_*_*', 
        'keep *_l1GtEmulDigis_*_*', 
        'keep *_l1GctDataDigis_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtDataEmul)
