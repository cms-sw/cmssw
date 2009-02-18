import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

readFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource", fileNames = readFiles)
readFiles.extend((
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/00969EA7-3E9C-DD11-8FC8-001617C3B654.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/02FCD49F-3E9C-DD11-A161-000423D98A44.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/0E0F669F-259C-DD11-9111-000423D991F0.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/1044DAA8-3E9C-DD11-A730-001617DBCF6A.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/18F416AF-3E9C-DD11-8F6E-000423D6CA02.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/2A80FAAA-3E9C-DD11-B1BD-001617E30CA4.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/2AEE68AB-3E9C-DD11-A40F-001617DBD224.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/3AA317AB-3E9C-DD11-972A-000423D9853C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/40CA2CE2-249C-DD11-9C52-001617E30D0A.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/42BCB6AE-3E9C-DD11-B97B-000423D6B358.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/4CD4F1A6-3E9C-DD11-9760-000423D99F3E.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/54C0FCA2-3E9C-DD11-9BA1-001617C3B710.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/5A45D0AD-3E9C-DD11-AF5D-000423D94A04.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/5E6638E5-249C-DD11-8AC8-000423D94494.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/608D46B3-249C-DD11-8AA5-0019DB2F3F9B.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/6E07E0A4-3E9C-DD11-A089-000423D99B3E.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/6E73FAAA-3E9C-DD11-9B53-000423D94534.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/70978FA9-3E9C-DD11-8D7E-000423D99160.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/740790A8-3E9C-DD11-831B-001617DBCF90.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/82507EA9-3E9C-DD11-ACC9-000423D98930.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/863655A2-3E9C-DD11-A40D-001617C3B77C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/88B4E7AD-3E9C-DD11-9814-000423D98804.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/8AEF0EA4-3E9C-DD11-95BC-000423D99660.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/900FFDAD-3E9C-DD11-B075-000423D9A212.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/9C82B1A7-3E9C-DD11-A426-000423D98EA8.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/A411CFAC-3E9C-DD11-A461-000423D9870C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/A87BF7AC-3E9C-DD11-9B97-001617C3B73A.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/B8DE3754-259C-DD11-A0B5-001617C3B6DE.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/C6DBDC2B-259C-DD11-9E2E-000423D9890C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/C843E8A1-3E9C-DD11-A66F-001617E30D38.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/CCDF81A4-3E9C-DD11-85B4-000423D8F63C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/CCF4ACAA-3E9C-DD11-8DB4-000423D99896.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/D069BFA6-3E9C-DD11-B080-000423D94A20.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/D8B9CC46-2F9C-DD11-8C5F-001617C3B78C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/DC88C2B0-249C-DD11-A1E2-001617DBCF6A.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/E48BC7A4-3E9C-DD11-AA7E-001617E30F50.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/E8322FA1-3E9C-DD11-9AD2-000423D98634.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/EE08B87E-249C-DD11-A4DB-000423D991D4.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/F6B2ABAF-3E9C-DD11-B339-000423D6A6F4.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/FEB5FBAE-3E9C-DD11-97B5-000423D944F8.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/637/FED569A3-3E9C-DD11-A75E-001617DBD5B2.root'
))

process.cscdumper = cms.EDAnalyzer("CSCFileDumper",
    output = cms.untracked.string("/tmp/kkotov/66637.bin"),
    events = cms.untracked.string("1073500,1166393")
)

process.p = cms.Path(process.cscdumper)

