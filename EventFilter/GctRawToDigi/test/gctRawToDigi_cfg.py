import FWCore.ParameterSet.Config as cms

process = cms.Process( "GctRawToDigi" )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(100)

process.dumpRaw = cms.OutputModule( "DumpFEDRawDataProduct",
  feds = cms.untracked.vint32( 745 ),
  dumpPayload = cms.untracked.bool( True )
)

# unpacker
process.load( "EventFilter/GctRawToDigi/l1GctHwDigis_cfi" )
process.l1GctHwDigis.inputLabel = cms.InputTag( "rawDataCollector" )
process.l1GctHwDigis.hltMode = cms.bool( True )
process.l1GctHwDigis.verbose = cms.untracked.bool( True )

# dump digis
process.load('L1Trigger.L1GctAnalyzer.dumpGctDigis_cfi')
process.dumpGctDigis.rawInput = cms.untracked.InputTag( "l1GctHwDigis" )
process.dumpGctDigis.emuRctInput = cms.untracked.InputTag( "rctDigis" )
process.dumpGctDigis.emuGctInput = cms.untracked.InputTag( "gctDigis" )
process.dumpGctDigis.doHardware = cms.untracked.bool ( True )
process.dumpGctDigis.doEmulated = cms.untracked.bool ( False )
process.dumpGctDigis.doRctEm = cms.untracked.bool ( True )
process.dumpGctDigis.doInternEm = cms.untracked.bool ( False )
process.dumpGctDigis.doEm = cms.untracked.bool ( True )
process.dumpGctDigis.doJets = cms.untracked.bool ( True )
process.dumpGctDigis.doEnergySums = cms.untracked.bool ( True )

process.p = cms.Path( 
#    process.dumpRaw 
    process.l1GctHwDigis
    # process.dumpGctDigis
 )

process.output = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring ( 
    "keep *",
#    "keep *_l1GctHwDigis_*_*",
#    "keep *_gctDigiToRaw_*_*"
  ),
  
  fileName = cms.untracked.string( "gctDigis.root" )

)
process.out = cms.EndPath( process.output )

# input
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1 ) )

process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring(
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/FEF2404B-6AD3-DE11-AF58-0019B9F72BFF.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/FEBBDD2C-32D3-DE11-B400-003048678098.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/FAF3C91A-51D3-DE11-ABFD-0030487A3C9A.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/F6897A24-46D3-DE11-BB1F-001617C3B6CC.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/EEC707FF-52D3-DE11-A4B1-000423D94A20.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/EE99A5FC-67D3-DE11-A646-000423D6BA18.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/ECCA99CC-4CD3-DE11-8FE6-001D09F253C0.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/EA9112F9-5FD3-DE11-AEB6-001D09F2AD7F.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/E899B5C2-5BD3-DE11-A050-001D09F28D54.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/E86A0E64-68D3-DE11-8909-000423D990CC.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/E8616DF1-39D3-DE11-9DEA-003048D375AA.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/E81667B5-4ED3-DE11-A38A-003048D2BED6.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/E2C46F28-67D3-DE11-A5F0-001D09F24F1F.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/E2B9AEB3-52D3-DE11-AA03-001D09F25456.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/DC4F2D74-47D3-DE11-8802-001D09F28F1B.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/DA50FAAC-60D3-DE11-BA9F-0019B9F709A4.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/D69D8696-33D3-DE11-8413-0030486730C6.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/D4D087E7-3CD3-DE11-8C87-0030487A18A4.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/D4C27D28-50D3-DE11-AB46-000423D991F0.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/D41993FC-41D3-DE11-9108-001D09F2516D.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/D27974A9-69D3-DE11-B313-0019B9F72BFF.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/D0DEBA11-5DD3-DE11-903C-001D09F28D54.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/C649CA22-45D3-DE11-84D1-001D09F24399.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/C647923B-3ED3-DE11-8FBB-001D09F24763.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/C457514D-55D3-DE11-95FE-001D09F24F1F.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/C028C572-44D3-DE11-9E5C-001D09F29114.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/B41BAF03-42D3-DE11-B4E3-0030487C5CFA.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/B247A836-64D3-DE11-8A0D-000423D996C8.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/B2028696-66D3-DE11-9361-001D09F24F1F.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/AADABA16-4BD3-DE11-AEB7-003048D2BE12.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/A8D8F539-3CD3-DE11-8384-001D09F27003.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/A4FDF10A-5DD3-DE11-84E7-0019B9F7312C.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/A47CD26A-57D3-DE11-B8CC-000423D60FF6.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/A2BCB4E2-32D3-DE11-8FEC-003048D37560.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/9CCB1F6B-59D3-DE11-9315-001D09F24FBA.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/9A8BB7B9-37D3-DE11-BEBD-001D09F290CE.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/927CBBBF-53D3-DE11-AA01-001D09F2AF1E.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/8EFD131B-4BD3-DE11-9A7C-003048D2C0F4.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/8C0739A0-43D3-DE11-A453-000423D6CA42.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/8AF60D81-46D3-DE11-B852-001D09F252DA.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/88F92AD3-53D3-DE11-A5C3-0030487D0D3A.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/82F556C0-35D3-DE11-99C-001D09F2983F.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/820AF02E-64D3-DE11-A05F-000423D986A8.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/7ECE87CC-64D3-DE11-862C-000423D987FC.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/7EAD91F3-36D3-DE11-9E6F-001D09F2462D.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/7AE69694-65D3-DE11-94E9-001D09F34488.root',
    '/store/data/BeamCommissioning09/Calo/RAW/v1/000/121/238/7A92A028-67D3-DE11-BDCE-000423D987E0.root'
  )
)

#process.source = cms.Source( "PoolSource",
#  fileNames = cms.untracked.vstring( "file:gctRaw.root" )
#)
