import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

# Database configuration
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR10_P_V4::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#--- SiPixelRawToDigi ---#
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = "source"
process.siPixelDigis.Timing = True
process.siPixelDigis.IncludeErrors = True

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/F4FE5D06-EB3B-DF11-8D16-0030487D0D3A.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/DEE7E9A9-FC3B-DF11-8BF8-001D09F23D1D.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/D01A79A5-FC3B-DF11-BC0F-001D09F23174.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/C0A5D466-EC3B-DF11-BDF4-000423D9A2AE.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/A8EAADF9-073C-DF11-B618-000423D94700.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/9CFF1A58-E83B-DF11-9AF2-0030487A322E.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/7E6AA362-083C-DF11-85EC-001D09F2516D.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/66CBD22E-003C-DF11-BD9E-001D09F2905B.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/5CA00C93-053C-DF11-924A-0030487A3C9A.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/32E33210-FE3B-DF11-A8D8-001D09F26C5C.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/1E2AA3C3-1A3C-DF11-AF37-0030487CD6D2.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/10CD830F-FA3B-DF11-997D-0030487C8E02.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/A2674AF9-173C-DF11-8166-000423D98DD4.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/86B38B5B-123C-DF11-8D59-0030487CD7C6.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/6688948C-0F3C-DF11-B0DB-000423D990CC.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/44947723-153C-DF11-915D-0030487C60AE.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/28C8B312-133C-DF11-9EF8-000423D6006E.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/1EDF60D5-0E3C-DF11-9245-0030487CD6DA.root',
        '/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/14349126-153C-DF11-A00B-0030487C608C.root'

        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/F2B88195-EFE5-DE11-A686-000423D9989E.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/D60BA528-F3E5-DE11-8F2F-0030487C6062.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/D4645697-EFE5-DE11-B856-000423D99AAA.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/C8525F6D-F2E5-DE11-B742-0030487D0D3A.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/C410736D-F2E5-DE11-9277-001D09F2441B.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/BE0712E4-F0E5-DE11-A09C-000423D99AAA.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/AADBCFE3-F0E5-DE11-BBBE-000423D9997E.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/A40B666D-F2E5-DE11-92EE-001D09F25401.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/86D313B7-F1E5-DE11-BE16-001D09F2525D.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/86799AEA-F0E5-DE11-AB11-001617E30D40.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/6C52EDE7-F0E5-DE11-9D0A-001617C3B66C.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/68FE5E6D-F2E5-DE11-AB8B-001D09F23174.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/5E103228-F3E5-DE11-A825-000423D98E6C.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/5641226F-F2E5-DE11-A753-001D09F2B30B.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/4CE5FA11-F1E5-DE11-8FCC-001D09F28D4A.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/382A8C13-F1E5-DE11-ADA0-000423D8F63C.root',
        #'/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/970/10A7B9CA-F8E5-DE11-840B-003048D3750A.root'
  )
)

##
## number of events
##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000) )

##
## executionpath
##
process.p = cms.Path(

    process.siPixelDigis

    )

process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.cerr.threshold = 'INFO'
