import FWCore.ParameterSet.Config as cms

process = cms.Process("alcarecoval")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet( ## kill all messages in the log
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet( ## but FwkJob category - those unlimitted
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('FwkJob'),
    destinations = cms.untracked.vstring('cout')
)

process.load("FWCore.MessageService.MessageLogger_cfi")



process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#'/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/ALCARECO/IDEAL_30X_StreamALCARECOHcalCalHO_v1/0001/0613F799-DB03-DE11-9C63-000423D6B48C.root'
# '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/58E14832-DB03-DE11-81B5-001617C3B79A.root'    
# '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_80_120/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/64EDD11D-DB03-DE11-97D9-000423D98B6C.root'
# '/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/ALCARECO/IDEAL_30X_StreamALCARECOHcalCalHO_v1/0001/0613F799-DB03-DE11-9C63-000423D6B48C.root'
 '/store/relval/CMSSW_3_1_0_pre2/RelValWM/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/FEC44C72-DB03-DE11-8D8C-000423D9A212.root'
# '/store/relval/CMSSW_3_1_0_pre2/RelValZMM/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/228907A2-DB03-DE11-B50A-000423D99F3E.root'


#'/store/relval/CMSSW_2_2_6/RelValTTbar/ALCARECO/STARTUP_V9_StreamALCARECOHcalCalHO_v1/0002/36E6ED60-490C-DE11-93D5-001D09F23A6B.root'
#'/store/relval/CMSSW_2_2_6/RelValQCD_Pt_3000_3500/ALCARECO/STARTUP_V9_StreamALCARECOHcalCalHO_v1/0002/5012E6A9-490C-DE11-8F5E-0030487A322E.root'
#'/store/relval/CMSSW_2_2_6/RelValQCD_Pt_80_120/ALCARECO/STARTUP_V9_StreamALCARECOHcalCalHO_v1/0002/BA3B4490-490C-DE11-ABD5-001D09F290BF.root'
#'/store/relval/CMSSW_2_2_6/RelValTTbar/ALCARECO/IDEAL_V12_StreamALCARECOHcalCalHO_v1/0002/BE79CBC6-490C-DE11-8824-0019B9F70607.root'
#'/store/relval/CMSSW_2_2_6/RelValWM/ALCARECO/STARTUP_V9_StreamALCARECOHcalCalHO_v1/0002/CA95EC87-490C-DE11-83E0-001D09F24FBA.root'
#'/store/relval/CMSSW_2_2_6/RelValZMM/ALCARECO/STARTUP_V9_StreamALCARECOHcalCalHO_v1/0002/0C9CCFED-490C-DE11-8049-0030487C6090.root'

#'/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/6C3F9560-7E0A-DE11-924E-001617E30CC8.root'
#'/store/relval/CMSSW_3_1_0_pre3/RelValQCD_Pt_3000_3500/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/B2B7B261-7E0A-DE11-AE47-000423D992A4.root'
#'/store/relval/CMSSW_3_1_0_pre3/RelValQCD_Pt_80_120/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/564C1855-7E0A-DE11-865D-000423D98634.root'
#'/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/ALCARECO/IDEAL_30X_StreamALCARECOHcalCalHO_v1/0001/1EB2A6C1-FA0A-DE11-8AD0-001D09F23D1D.root',
#        '/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/ALCARECO/IDEAL_30X_StreamALCARECOHcalCalHO_v1/0001/94485045-800A-DE11-B5A4-001617E30F50.root'
#'/store/relval/CMSSW_3_1_0_pre3/RelValWM/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/6EE937C7-7D0A-DE11-8380-001617E30D12.root'
#'/store/relval/CMSSW_3_1_0_pre3/RelValZMM/ALCARECO/STARTUP_30X_StreamALCARECOHcalCalHO_v1/0001/968A4F18-7E0A-DE11-914F-001617DBD332.root'
))

process.load("DQMServices.Core.DQMStore_cfg")


process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.o1 = cms.OutputModule("PoolOutputModule",
   outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('event_content_relvalsinglemupt1000.root')
)

process.load("DQMOffline.CalibCalo.MonitorHOAlCaRecoStream_cfi")
process.MonitorHOAlCaRecoStream.nbins=40
process.MonitorHOAlCaRecoStream.highedge=6.0

#process.p = cms.Path(process.dump*process.MonitorHOAlCaReco)
process.p = cms.Path(process.MonitorHOAlCaRecoStream)
#process.p = cms.Path(process.dump);
#process.e = cms.EndPath(process.o1)
