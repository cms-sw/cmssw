import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_8_0/MinimumBias/ALCARECO/GR_R_38X_V7_RelVal_col_10_bs_TkAlMinBias-v1/0007/90B6F8D6-4E96-DF11-B3A9-0018F3D0970A.root',
'/store/relval/CMSSW_3_8_0/MinimumBias/ALCARECO/GR_R_38X_V7_RelVal_col_10_bs_TkAlMinBias-v1/0007/8A6CDD40-C995-DF11-B6D5-0018F3D096AA.root'
#'/store/relval/CMSSW_3_8_0/MinimumBias/ALCARECO/GR_R_38X_V7_RelVal_col_10_bs_PromptCalibProd-v1/0007/783224D4-4E96-DF11-973F-00261894389F.root'
#'file:/uscms_data/d2/uplegger/CMSSW/CMSSW_3_8_0_pre7/src/Calibration/TkAlCaRecoProducers/test/AlcaBeamSpot100000.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_26_1_wr7.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_27_1_lLj.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_28_1_bnD.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_29_1_RMo.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_30_1_tRZ.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_31_1_K5H.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_32_1_A79.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_33_1_q4g.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_34_1_oQj.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_35_1_HiH.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_36_1_kov.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_37_1_uzV.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_38_1_1Fi.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_39_1_7S6.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_40_1_r55.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_41_1_fyj.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_42_1_O14.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_43_1_8Ro.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_44_1_7A2.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_45_1_pmd.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_46_1_US3.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_47_1_sJM.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_48_1_NT7.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_49_1_ojG.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_50_1_ws1.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_51_1_bmE.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_52_1_8YT.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_53_1_zd1.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_54_1_5ZX.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_55_1_FND.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_56_1_vQ7.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_57_1_RU4.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_58_1_Y3g.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_59_1_heB.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_60_1_V9K.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_61_1_arM.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_62_1_knX.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_63_1_Uuu.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_64_1_GVx.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_65_1_Tyj.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_66_1_VCk.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_67_1_OBg.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_68_1_l3C.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_69_1_Efy.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_70_1_CUT.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_71_1_zxy.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_72_1_Ic1.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_73_1_qFm.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_74_1_EYe.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_75_1_C7L.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_76_1_wHh.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_77_1_HaU.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_78_1_Rd6.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_79_1_acr.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_80_1_2yF.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_81_1_kO4.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_82_1_CHC.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_83_1_kg7.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_84_1_0NQ.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_85_1_qPn.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_86_1_JWP.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_87_1_FFq.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_88_1_dph.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_89_1_vFd.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_90_1_6fa.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_91_1_UnF.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_92_1_Whj.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_93_1_0kN.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_94_1_hmd.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_95_1_1hk.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_96_1_tyw.root',
#'file:/uscms/home/uplegger/nobackup/rootFiles/AlcaProducer/AlcaBeamSpot_97_1_3P8.root'
    )
#    , duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
#    , processingMode = cms.untracked.string('RunsAndLumis')

)

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories = ["AlcaBeamMonitor"]
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(0)
    ),
    AlcaBeamMonitor = cms.untracked.PSet(
        #reportEvery = cms.untracked.int32(100) # every 1000th only
	limit = cms.untracked.int32(10000)
    )
)
#process.MessageLogger.statistics.append('cout')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.maxLuminosityBlocks=cms.untracked.PSet(
#         input=cms.untracked.int32(1000)
#)

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.BeamMonitor.AlcaBeamMonitor_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    								   record = cms.string('BeamSpotObjectsRcd'),			        
#    								   tag = cms.string('BeamSpotObjects_2009_LumiBased_SigmaZ_v14_offline') 
    								   tag = cms.string('BeamSpotObject_ByLumi') 
    								  )
						         ),								        
                                         #connect = cms.string('frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT')
                                         connect = cms.string('sqlite_file:step4.db')
                                         #connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
                                         #connect = cms.string('frontier://PromptProd/CMS_COND_31X_BEAMSPOT')
                                        )



process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc08.fnal.gov'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/DQM/TkAlCalibration/ALCARECO'
process.dqmEnv.subSystemFolder = 'AlcaBeamMonitor'
process.dqmSaver.saveByRun = 1
#process.dqmSaver.saveAtJobEnd = True

#import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
#process.offlineBeamSpotForDQM = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.pp = cms.Path(process.AlcaBeamMonitor+process.dqmSaver)
process.schedule = cms.Schedule(process.pp)





#process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
#process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124120:1-124120:59')

# this is for filtering on L1 technical trigger bit
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
#process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
#process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
#process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 )')

#### remove beam scraping events
#process.noScraping= cms.EDFilter("FilterOutScraping",
#    applyfilter = cms.untracked.bool(True),
#    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
#    numtrack = cms.untracked.uint32(10),
#    thresh = cms.untracked.double(0.20)
#)

#process.dqmBeamMonitor.Debug = True
#process.dqmBeamMonitor.BeamFitter.Debug = True
#process.dqmBeamMonitor.BeamFitter.WriteAscii = True
#process.dqmBeamMonitor.BeamFitter.AsciiFileName = 'BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
#process.dqmBeamMonitor.BeamFitter.DIPFileName = 'BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = True
#process.dqmBeamMonitor.BeamFitter.OutputFileName = 'BeamFitResults.root'
#process.dqmBeamMonitor.resetEveryNLumi = 10
#process.dqmBeamMonitor.resetPVEveryNLumi = 5

## bx
#process.dqmBeamMonitorBx.Debug = True
#process.dqmBeamMonitorBx.BeamFitter.Debug = True
#process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
#process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = 'BeamFitResultsBx.txt'

### TKStatus
#process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
#	BeamFitter = cms.PSet(
#	DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
#	)
#)
###

#process.pp = cms.Path(process.dqmTKStatus*process.hltLevel1GTSeed*process.dqmBeamMonitor+process.dqmBeamMonitorBx+process.dqmEnv+process.dqmSaver)
#process.dqmSaver.dirName = '.'
#process.dqmSaver.producer = 'Playback'
#process.dqmSaver.convention = 'Online'
#process.dqmEnv.subSystemFolder = 'BeamMonitor'
#process.dqmSaver.saveByRun = 1
#process.dqmSaver.saveAtJobEnd = True


#process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
#                                        process.CondDBSetup,
#                                        toGet = cms.VPSet(cms.PSet(
#    record = cms.string('BeamSpotObjectsRcd'),
#    tag = cms.string('Early10TeVCollision_3p8cm_v3_mc_IDEAL')
#    )),
#    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
#    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_BEAMSPOT')
#)


