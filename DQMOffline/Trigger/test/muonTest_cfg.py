import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

##### Templates to change parameters in muonTriggerRateTimeAnalyzer
# process.muonTriggerRateTimeAnalyzer.NtupleFileName = cms.untracked.string("ntuple.root")
# process.muonTriggerRateTimeAnalyzer.TriggerNames = cms.vstring("HLT_IsoMu9")
# process.muonTriggerRateTimeAnalyzer.MinPtCut = cms.untracked.double(10.)
# process.muonTriggerRateTimeAnalyzer.MotherParticleId = cms.untracked.uint32(24)
# process.muonTriggerRateTimeAnalyzer.HltProcessName = cms.string("HLT2")
# process.muonTriggerRateTimeAnalyzer.UseAod = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

#process.source = cms.Source("PoolSource",
#    skipEvents = cms.untracked.uint32(0),
#	# Events with GEN-SIM-HLTDEBUG						
#    #fileNames  = cms.untracked.vstring('/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/CC5DE776-6DF3-DD11-B62A-001617C3B5E4.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/1075C498-6DF3-DD11-9217-001D09F34488.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/C4E1E564-69F3-DD11-863B-001D09F251FE.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/A25C241A-6DF3-DD11-9CFE-001617C3B778.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/1E183450-6DF3-DD11-8764-000423D990CC.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/9660C722-6AF3-DD11-A15D-000423D99F1E.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/74B2C4EB-69F3-DD11-80AC-000423D6CA72.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0621AFE7-69F3-DD11-8E0F-001D09F23A07.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/B60F178C-69F3-DD11-BC41-001D09F2A465.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/F4D741FB-69F3-DD11-B311-001D09F251B8.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/E8377D2A-6AF3-DD11-B1CA-001D09F23174.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/C8846308-6AF3-DD11-89BF-000423D94990.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/F81C1FF9-90F3-DD11-82B1-001D09F2A465.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0001/C07A4356-C6F3-DD11-A378-0019B9F707D8.root')
#	fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/060A7A48-69F3-DD11-9D9D-000423D98E54.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/16E25010-6AF3-DD11-855D-000423D98920.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/28759004-91F3-DD11-B44B-001D09F2525D.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/B6CBA475-6DF3-DD11-8ADC-000423D98804.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0001/EC94B557-C6F3-DD11-BDCE-0019B9F70607.root')
#
#)


process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
	# Events with GEN-SIM-HLTDEBUG						
    #fileNames  = cms.untracked.vstring('/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/CC5DE776-6DF3-DD11-B62A-001617C3B5E4.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/1075C498-6DF3-DD11-9217-001D09F34488.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/C4E1E564-69F3-DD11-863B-001D09F251FE.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/A25C241A-6DF3-DD11-9CFE-001617C3B778.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/1E183450-6DF3-DD11-8764-000423D990CC.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/9660C722-6AF3-DD11-A15D-000423D99F1E.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/74B2C4EB-69F3-DD11-80AC-000423D6CA72.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0621AFE7-69F3-DD11-8E0F-001D09F23A07.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/B60F178C-69F3-DD11-BC41-001D09F2A465.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/F4D741FB-69F3-DD11-B311-001D09F251B8.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/E8377D2A-6AF3-DD11-B1CA-001D09F23174.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/C8846308-6AF3-DD11-89BF-000423D94990.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/F81C1FF9-90F3-DD11-82B1-001D09F2A465.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0001/C07A4356-C6F3-DD11-A378-0019B9F707D8.root')
	# older gen-sim-reco						
	# fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/060A7A48-69F3-DD11-9D9D-000423D98E54.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/16E25010-6AF3-DD11-855D-000423D98920.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/28759004-91F3-DD11-B44B-001D09F2525D.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/B6CBA475-6DF3-DD11-8ADC-000423D98804.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0001/EC94B557-C6F3-DD11-BDCE-0019B9F70607.root')
	# new gen-sim-reco						
#	fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/44CFC665-57E8-DD11-B3EC-000423D6B5C4.root', '/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/7C3E2A10-6BE8-DD11-A4C8-000423D944F0.root', '/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/E89CF1EB-57E8-DD11-B14B-000423D6CA02.root', '/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/EC6705D2-57E8-DD11-9009-000423D98E6C.root')
fileNames = cms.untracked.vstring('file:test.root')
)


#
#  Code from the samples page to use ZMM samples with GEN-SIM-RECO
#  See if we can do AOD matching 
#
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.readFiles = cms.untracked.vstring()
#process.secFiles = cms.untracked.vstring() 
#process.source = cms.Source ("PoolSource",fileNames = process.readFiles, secondaryFileNames = process.secFiles)
#process.readFiles.extend( [
#       '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/060A7A48-69F3-DD11-9D9D-000423D98E54.root',
#       '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/16E25010-6AF3-DD11-855D-000423D98920.root',
#       '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/28759004-91F3-DD11-B44B-001D09F2525D.root',
#       '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/B6CBA475-6DF3-DD11-8ADC-000423D98804.root',
#       '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0001/EC94B557-C6F3-DD11-BDCE-0019B9F70607.root' ] );
#process.secFiles.extend( [
#               ] )
#


process.DQMStore = cms.Service("DQMStore")

process.MessageLogger = cms.Service("MessageLogger",
    HLTMuonVallog  = cms.untracked.PSet(
        threshold  = cms.untracked.string('INFO'),
        default    = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HLTMuonVal = cms.untracked.PSet(
            limit = cms.untracked.int32(100000)
        )
    ),
    debugModules   = cms.untracked.vstring('*'),
    cout           = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG')
    ),
    categories     = cms.untracked.vstring('HLTMuonVal'),
    destinations   = cms.untracked.vstring('cout', 'HLTMuonVal.log')
)

# JMS try not to throw out branches w/ histos
#process.out = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring(
#        'drop *', 
#        'keep *_MEtoEDMConverter_*_HLTMuonOfflineAnalysis'),
#    fileName = cms.untracked.string('RateTimeAnalyzer_n50_useAod.root')
#)

process.out = cms.OutputModule("PoolOutputModule",
	 outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
	 fileName = cms.untracked.string('RateTimeAnalyzer_n5000_useAod_RECO.root')
)

# JMS try to remove MEtoEDM step
process.analyzerpath = cms.Path(
    process.offlineDQMMuonTrig*
    process.MEtoEDMConverter
)

#process.analyzerpath = cms.Path(
#    process.muonTriggerRateTimeAnalyzer
#)

process.outpath = cms.EndPath(process.out)
