import FWCore.ParameterSet.Config as cms

process = cms.Process("newtest")

from Geometry.CaloEventSetup.CaloTopology_cfi import *

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")



process.load("DQMOffline.CalibCalo.MonitorAlCaEcalPi0_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/0A984025-9316-DE11-AC6D-001731EF61B4.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/2E8D4B82-A316-DE11-A903-001A92811748.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/3406EA9B-9E16-DE11-8FA0-001A9281171E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/46CAE5DE-9316-DE11-BBE0-003048678E6E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/480ABE32-9E16-DE11-B0BC-0030486792DE.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/48CCA350-A116-DE11-8496-0018F3D096EC.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/4A395DFF-A116-DE11-A191-0018F3D09650.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/4AB0335B-9416-DE11-933B-003048678B7E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/5C40780E-A016-DE11-9CA3-001A9281171E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/5E289786-9216-DE11-B404-0018F3D096D2.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/608519C7-A216-DE11-8925-0018F3D09670.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/70B58652-A116-DE11-8898-0018F3D0969C.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/7CC4C224-A016-DE11-B98F-003048678F8E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/7E2C9D76-A516-DE11-8C3B-0030486792B8.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/80BEE5B0-A216-DE11-A64C-0018F3D0961A.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/80FD358A-A316-DE11-9678-00304876A065.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/928683B1-A216-DE11-BF43-001A92810AEA.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/AAF6E25A-9416-DE11-8DE7-003048679070.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/B2C1EBDF-9216-DE11-9E23-0018F3D096C6.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/B42EB9E3-9316-DE11-BF02-00304867916E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/BA0BE15A-9416-DE11-8D30-003048D15E14.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/C2FB121A-9716-DE11-8B91-0018F3D096FE.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/CED7ECC2-9516-DE11-8C83-0018F3D0968A.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/D4C28B19-A016-DE11-B2F6-001A92810AEE.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/DAAB1FA5-A016-DE11-9D68-001A92971BCA.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/E488C6CA-9216-DE11-8976-003048678FB4.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/E4C9C18E-9216-DE11-AA62-003048678A7E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/EA9C19A5-A016-DE11-A24F-001A92971BCA.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/F8F79950-9316-DE11-AE36-001731AF6847.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/2601C9D0-D416-DE11-B162-001BFCDBD160.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/E8871A36-D016-DE11-A62D-003048679228.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/04645849-1416-DE11-ABA4-0016177CA778.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/061B4A6A-1116-DE11-B673-001617C3B6C6.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/0661E9C1-1016-DE11-A731-000423D9A2AE.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/1000215D-1216-DE11-9082-001617E30D52.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/1064C461-1016-DE11-B2B4-000423D99F3E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/4E9B9893-1216-DE11-884A-001617C3B76A.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/546C7780-1A16-DE11-8DD3-000423D94494.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/621A95C9-5116-DE11-977A-000423D98DD4.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/6A784C87-1216-DE11-BD50-0019DB29C5FC.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/8286D6B6-1416-DE11-A666-000423D98FBC.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/9EA5B16A-0E16-DE11-BE09-000423D9A2AE.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/B0F104EA-0F16-DE11-93B8-000423D996C8.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/BE22BC09-0E16-DE11-B249-001617C3B6C6.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/C85A027F-5116-DE11-BD00-000423D98EC4.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/D096AEE7-1616-DE11-BED6-0016177CA778.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/D843A533-AB16-DE11-8802-000423D6CA72.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/FCAD1305-5116-DE11-BBB6-001617E30D38.root'
        
        
)
)

process.out1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *'),
    fileName = cms.untracked.string('dqm.root')
)

process.p = cms.Path(process.EcalPi0Mon*process.MEtoEDMConverter)
process.o = cms.EndPath(process.out1)
process.EcalPi0Mon.SaveToFile = True
process.MEtoEDMConverter.Verbosity = 1


