import FWCore.ParameterSet.Config as cms

process = cms.Process("HITval")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000) )

process.source = cms.Source("PoolSource",
    # put here 10 TeV RelVal minbias sample with HLTDEBUG, RAW & RECO
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/184A9883-DC15-DE11-AD56-000423D94700.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/3AF613E5-DB15-DE11-8D0B-001617C3B6E8.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/50E7F2AC-DD15-DE11-80B6-001617C3B70E.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/64EF6304-DB15-DE11-9930-000423D9A212.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/742D8D4C-DF15-DE11-B820-000423D98B6C.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/784587EC-DD15-DE11-8A23-001617C3B79A.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/78BC157B-DB15-DE11-A112-000423D9870C.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/7C8347C2-5016-DE11-A08B-001617E30CC8.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/86E1820C-E615-DE11-B29A-001617C3B73A.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/9CABEC94-E115-DE11-A9C7-000423D94990.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/DC789C0B-D615-DE11-90CE-000423D944FC.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/DE497988-D615-DE11-8D94-000423D98B5C.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/E652FA2F-AB16-DE11-87F9-000423D992DC.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/EC95492A-DD15-DE11-8AF7-001617DBD288.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/EE498124-F515-DE11-BA4D-000423D944F8.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/F67E56F1-D915-DE11-A59C-000423D98920.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/FAC61CE9-DE15-DE11-B797-001617E30D0A.root'
)
)

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

######### select L1 menu for which you want to validate/evaluate rates
# in L1TriggerConfig.L1GtConfigProducers.Luminosity
process.load('Configuration/StandardSequences/L1TriggerDefaultMenu_cff')

process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")
process.hltHighLevel.HLTPaths = ['HLT_IsoTrack']

process.load("HLTriggerOffline.special.hltHITval_cfi")
process.hltHITval.doL1Prescaling=cms.bool(True)
process.hltHITval.produceRates=cms.bool(True)
process.hltHITval.luminosity=cms.double(8e29)
process.hltHITval.sampleCrossSection=cms.double(7.53E10)
process.hltHITval.hltL3FilterLabel=cms.InputTag("hltIsolPixelTrackFilter::HLT")
process.hltHITval.SaveToRootFile=cms.bool(True)

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.dqmOut = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('dqmHltHITval_minBias.root'),
     outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
 )

process.p = cms.Path(process.hltHITval+process.MEtoEDMConverter)

process.ep=cms.EndPath(process.dqmOut)
