import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# this will create a list of producers
# each producer provides a new muon collection
process.load("DQMOffline.Trigger.muonSelector_cfi")

##### Templates to change parameters in muonTriggerRateTimeAnalyzer
# process.muonTriggerRateTimeAnalyzer.NtupleFileName = cms.untracked.string("ntuple.root")
# process.muonTriggerRateTimeAnalyzer.TriggerNames = cms.vstring("HLT_IsoMu9")
# process.muonTriggerRateTimeAnalyzer.MinPtCut = cms.untracked.double(10.)
# process.muonTriggerRateTimeAnalyzer.MotherParticleId = cms.untracked.uint32(24)
# process.muonTriggerRateTimeAnalyzer.HltProcessName = cms.string("HLT2")
# process.muonTriggerRateTimeAnalyzer.UseAod = cms.untracked.bool(True)

#------- Update: Use this format instead
#process.offlineDQMMuonTrig.BlahBlah

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(8000)
	input = cms.untracked.int32(5)
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
	#fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/060A7A48-69F3-DD11-9D9D-000423D98E54.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/16E25010-6AF3-DD11-855D-000423D98920.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/28759004-91F3-DD11-B44B-001D09F2525D.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0000/B6CBA475-6DF3-DD11-8ADC-000423D98804.root', '/store/relval/CMSSW_2_2_4/RelValZMM/GEN-SIM-RECO/STARTUP_V8_v1/0001/EC94B557-C6F3-DD11-BDCE-0019B9F70607.root')

	# new gen-sim-reco
							# has eta-phi bug
							#fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/44CFC665-57E8-DD11-B3EC-000423D6B5C4.root', '/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/7C3E2A10-6BE8-DD11-A4C8-000423D944F0.root', '/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/E89CF1EB-57E8-DD11-B14B-000423D6CA02.root', '/store/relval/CMSSW_3_0_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0006/EC6705D2-57E8-DD11-9009-000423D98E6C.root')

							# new gen-sim-reco
							# doesn't have geometry bug
							# 3_1_0_pre1
							#fileNames = cms.untracked.vstring ( '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0001/1CE757BC-06F8-DD11-94B7-000423D987FC.root',
							#									'/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0001/42EDD21F-F7F7-DD11-A8F4-0030487A322E.root',
							#									'/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0001/72C1C406-F7F7-DD11-9D12-0030487A3C9A.root',
							#									'/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0001/D63EA9DC-F6F7-DD11-85DC-000423D99658.root')

							fileNames = cms.untracked.vstring ('/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/5C786CEA-D415-DE11-9F1D-000423D6B358.root',
															   '/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/62F9AF48-E315-DE11-AF8E-001D09F24047.root',
															   '/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/762338D4-6316-DE11-8D10-000423D991F0.root',
															   '/store/relval/CMSSW_3_1_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP_30X_v1/0003/AEDC409C-AB16-DE11-BE1D-001617E30E28.root'  ),


							# try GEN-SIM-DIGI-RAW-HLTDEBUG
							#fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/02695F20-F7F7-DD11-AD0B-001617C3B78C.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/109EF1BD-06F8-DD11-A089-000423D98930.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/1C739DCD-F6F7-DD11-AF83-000423D996B4.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/22EFFAD9-F6F7-DD11-8DBB-001617E30D4A.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/38DD7BC3-F6F7-DD11-B5DB-000423D98BE8.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/58771E07-F7F7-DD11-9793-001D09F2A465.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/7C212906-F7F7-DD11-BFF8-001617DBD224.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/809BEE12-F7F7-DD11-9AC5-001617C3B6DC.root',
							#							  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/82E7D12C-F7F7-DD11-9CD8-001617E30E2C.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/C2739704-F7F7-DD11-8FEB-0030487D1BCC.root',
							#								  '/store/relval/CMSSW_3_1_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/D4637407-F7F7-DD11-8D4D-00304879FA4A.root')



							
# new gen-sim-reco
							# 
							

    #fileNames = cms.untracked.vstring('file:test.root')
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
	# Be careful - this can print a lot of debug info
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
	 fileName = cms.untracked.string('/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_n10_useAodAndRAW_vMorePlots.root')
)

# Path must contain all producers
# Don't use the other producers,
# keep them out of the path for now
process.analyzerpath = cms.Path(	
    process.offlineDQMMuonTrig*
    process.MEtoEDMConverter
)


#	process.highPtMuonTracks*
	#	process.externalMuonTracks*
	#   process.endcapMuonTracks*
	#process.barrelMuonTracks*
	#process.overlapMuonTracks*	

# This is the simplified version
# use in conjunction with  the
# collection name "globalMuons"
#
# process.analyzerpath = cms.Path (
#	 process.offlineDQMMuonTrig*
#	 process.MEtoEDMConverter
#	 )

#process.analyzerpath = cms.Path(
#    process.muonTriggerRateTimeAnalyzer
#)

process.outpath = cms.EndPath(process.out)
