import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# DQM services
#process.load("DQMServices.Core.DQM_cfg")

# Database configuration
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.load("CondCore.DBCommon.CondDBSetup_cfi")

# conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_52_V2::All"
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#--- SiPixelRawToDigi ---#
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
#process.siPixelDigis.InputLabel = "rawDataCollector"
process.siPixelDigis.InputLabel = "rawDataCollector"
process.siPixelDigis.Timing = True
process.siPixelDigis.UseQualityInfo = False
process.siPixelDigis.IncludeErrors = True
process.siPixelDigis.ErrorList = [29]
process.siPixelDigis.UserErrorList = [40]

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0046/2EC30434-4DD4-DF11-A881-003048D3FC94.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/DABA1B6F-22D2-DF11-B74F-002618943811.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/D0FA406E-22D2-DF11-8726-003048678AF4.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/C8D6E572-22D2-DF11-A66E-001A92811710.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/C27E3445-23D2-DF11-8B4B-002618943961.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/BE45905B-E9D1-DF11-AB7C-001A92810AE0.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/B27ECC44-23D2-DF11-B179-002618943974.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/B2183235-E8D1-DF11-9939-002618943932.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/8C97336E-22D2-DF11-A904-00304867902C.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/8C8C5E71-22D2-DF11-9E77-002618943838.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/7EABF0A2-25D2-DF11-B644-0018F3D09700.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/74E41B31-E8D1-DF11-B7EE-0018F3D09702.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/225A546D-22D2-DF11-BF6C-003048678F06.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/0A05BC4B-EAD1-DF11-8A6F-003048678F8E.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/0249DC70-22D2-DF11-A6F5-003048678F26.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/00297A4B-EAD1-DF11-A1D6-002618943913.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0039/EC7E8E21-E6D1-DF11-99D0-002354EF3BE0.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0039/EA7B3392-E4D1-DF11-A171-002354EF3BDC.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0039/AE8DDE1D-E7D1-DF11-B0F5-002618943877.root',
        #'/store/relval/CMSSW_3_8_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0039/70803B9E-E6D1-DF11-91FE-0026189438F3.root'

#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/F4FE5D06-EB3B-DF11-8D16-0030487D0D3A.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/DEE7E9A9-FC3B-DF11-8BF8-001D09F23D1D.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/D01A79A5-FC3B-DF11-BC0F-001D09F23174.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/C0A5D466-EC3B-DF11-BDF4-000423D9A2AE.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/A8EAADF9-073C-DF11-B618-000423D94700.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/9CFF1A58-E83B-DF11-9AF2-0030487A322E.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/7E6AA362-083C-DF11-85EC-001D09F2516D.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/66CBD22E-003C-DF11-BD9E-001D09F2905B.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/5CA00C93-053C-DF11-924A-0030487A3C9A.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/32E33210-FE3B-DF11-A8D8-001D09F26C5C.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/1E2AA3C3-1A3C-DF11-AF37-0030487CD6D2.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/440/10CD830F-FA3B-DF11-997D-0030487C8E02.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/A2674AF9-173C-DF11-8166-000423D98DD4.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/86B38B5B-123C-DF11-8D59-0030487CD7C6.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/6688948C-0F3C-DF11-B0DB-000423D990CC.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/44947723-153C-DF11-915D-0030487C60AE.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/28C8B312-133C-DF11-9EF8-000423D6006E.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/1EDF60D5-0E3C-DF11-9245-0030487CD6DA.root',
#        'rfio:/castor/cern.ch/cms/store/data/Commissioning10/ZeroBias/RAW/v4/000/132/442/14349126-153C-DF11-A00B-0030487C608C.root'

        'file:/tmp/andrewdc/CMSSW_5_2_X_2012-04-06-0100/src/Fall11_TTbarZIncl_TuneZ2_7TeV-madgraph-tauola_GEN-RAW.root'
  )
)

##
## number of events
##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100) )

##
## Output
##
process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:digis.root'),
    outputCommands = cms.untracked.vstring("drop *","keep *_siPixelDigis_*_*")
)
##
## executionpath
##
process.Timing = cms.Service("Timing")
process.p = cms.Path(process.siPixelDigis)
process.ep = cms.EndPath(process.out)

process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.cerr.threshold = 'INFO'
