import FWCore.ParameterSet.Config as cms

process = cms.Process( "GctRawToDigi" )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service ( "MessageLogger",
#  destinations = cms.untracked.vstring ( "debug.log" ),
#  debug = cms.untracked.PSet ( threshold = cms.untracked.string ( "DEBUG" ) ),
#  debugModules = cms.untracked.vstring ( "DumpFedRawDataProduct", "TextToRaw", "GctRawToDigi" )
#)

process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/FA7998E0-3999-DE11-ABD0-00304867920C.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/FA6C82EF-3999-DE11-89FA-001731AF6BD3.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F8982D55-2299-DE11-B6D6-001731AF687F.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F6A44E75-2299-DE11-9537-0018F3D09678.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F67D8909-CB98-DE11-8A97-003048678B86.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F42FFA84-CA98-DE11-AB82-0018F3D0961A.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F296F67F-2299-DE11-B414-001A92971B38.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/EED22D89-2299-DE11-9CBD-001A92810AEA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/EE0E8370-2299-DE11-8FD0-0018F3D0967E.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/EAD24A70-2299-DE11-A577-0018F3D0961A.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/E6EE64EF-3999-DE11-B049-001731AF66AF.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/E6D76B53-2299-DE11-B42D-001731AF6BC1.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/DEDBFB81-9F98-DE11-94B8-0018F3D0967E.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/DCFADA7F-2299-DE11-8B70-001731AF66EF.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D8CE2597-2299-DE11-94FE-0018F3D096E6.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D892C2B1-CA98-DE11-A5A9-001A92971AA8.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D88E3C61-2299-DE11-824F-0018F3D096BA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D4FF6838-CB98-DE11-BE10-0017312310E7.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D452C271-9F98-DE11-A96F-001731AF6765.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D2056A6D-2299-DE11-80C6-001A92971BDA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D0BC3269-9F98-DE11-B240-0018F3D096CA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/CCF49F7B-2299-DE11-9256-0018F3D0966C.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/CAE32973-2299-DE11-AE58-001A92810AEE.root'
  )
)

#process.source = cms.Source( "PoolSource",
#  fileNames = cms.untracked.vstring( "file:gctRaw.root" )
#)

process.dumpRaw = cms.OutputModule( "DumpFEDRawDataProduct",
  feds = cms.untracked.vint32( 745 ),
  dumpPayload = cms.untracked.bool( True )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1000 ) )

process.load( "EventFilter/GctRawToDigi/l1GctHwDigis_cfi" )
process.l1GctHwDigis.inputLabel = cms.InputTag( "source" )

process.p = cms.Path( process.dumpRaw * process.l1GctHwDigis )

process.output = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring ( 
    "keep *",
#    "keep *_l1GctHwDigis_*_*",
#    "keep *_gctDigiToRaw_*_*"
  ),
  
  fileName = cms.untracked.string( "gctDigis.root" )

)
process.out = cms.EndPath( process.output )
