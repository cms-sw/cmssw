import FWCore.ParameterSet.Config as cms


process = cms.Process("makeSD")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('SD and central skims'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Skimming/test/SDmaker_6SD_3CS_PDMinBias_1e28_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_R_36X_V11A::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F85204EE-EB40-DF11-8F71-001A64789D1C.root'
        ),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/F6887FD0-9371-DE11-B69E-00304879FBB2.root'
        )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

import HLTrigger.HLTfilters.hltHighLevelDev_cfi


### JetMETTau SD
process.JetMETTau_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.JetMETTau_1e28.HLTPaths = (
"HLT_Jet15U",
"HLT_DiJetAve15U_8E29",
"HLT_FwdJet20U",
"HLT_Jet30U", 
"HLT_Jet50U",
"HLT_DiJetAve30U_8E29",
"HLT_QuadJet15U",
"HLT_MET45",
"HLT_MET100",
"HLT_HT100U",
"HLT_SingleLooseIsoTau20",
"HLT_DoubleLooseIsoTau15",
"HLT_DoubleJet15U_ForwardBackward",
"HLT_BTagMu_Jet10U",
"HLT_BTagIP_Jet50U",
"HLT_StoppedHSCP_8E29"
)
process.JetMETTau_1e28.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
process.JetMETTau_1e28.HLTOverallPrescale = cms.uint32(1)
process.JetMETTau_1e28.throw = False
process.JetMETTau_1e28.andOr = True

process.filterSdJetMETTau_1e28 = cms.Path(process.JetMETTau_1e28)



### JetMETTauMonitor SD
process.JetMETTauMonitor_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.JetMETTauMonitor_1e28.HLTPaths = (
"HLT_L1Jet6U",
"HLT_L1MET20",
"HLT_L1SingleCenJet",
"HLT_L1SingleForJet",
"HLT_L1SingleTauJet",
"HLT_L1Jet10U",
"HLT_L1Jet10U_NoBPTX",
"HLT_L1Jet6U_NoBPTX",
"HLT_L1SingleCenJet_NoBPTX",
"HLT_L1SingleForJet_NoBPTX",
"HLT_L1SingleTauJet_NoBPTX"
)
process.JetMETTauMonitor_1e28.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1)
process.JetMETTauMonitor_1e28.HLTOverallPrescale = cms.uint32(1)
process.JetMETTauMonitor_1e28.throw = False
process.JetMETTauMonitor_1e28.andOr = True

process.filterSdJetMETTauMonitor_1e28 = cms.Path(process.JetMETTauMonitor_1e28)

### MuMonitor SD
process.MuMonitor_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.MuMonitor_1e28.HLTPaths = (
"HLT_L1MuOpen",
"HLT_L1Mu"
)
process.MuMonitor_1e28.HLTPathsPrescales  = cms.vuint32(1,1)
process.MuMonitor_1e28.HLTOverallPrescale = cms.uint32(1)
process.MuMonitor_1e28.throw = False
process.MuMonitor_1e28.andOr = True

process.filterSdMuMonitor_1e28 = cms.Path(process.MuMonitor_1e28)



### Mu SD
process.Mu_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.Mu_1e28.HLTPaths = (
"HLT_L2Mu0",
"HLT_L2Mu3",
#"HLT_L2Mu5",
"HLT_L1Mu20",
"HLT_L2Mu9",
"HLT_L2Mu11",
"HLT_L1Mu14_L1SingleEG10",
"HLT_L1Mu14_L1SingleJet6U",
"HLT_L1Mu14_L1ETM30",
"HLT_L2DoubleMu0",
"HLT_L1DoubleMuOpen",
"HLT_DoubleMu0",
"HLT_DoubleMu3",
"HLT_Mu3",
"HLT_Mu5",
"HLT_Mu9",
"HLT_IsoMu3",
"HLT_Mu0_L1MuOpen",
"HLT_Mu0_Track0_Jpsi",
"HLT_Mu3_L1MuOpen",
"HLT_Mu3_Track0_Jpsi",
"HLT_Mu5_L1MuOpen",
"HLT_Mu5_Track0_Jpsi",
"HLT_Mu0_L2Mu0",
"HLT_Mu3_L2Mu0",
"HLT_Mu5_L2Mu0"
)
process.Mu_1e28.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
process.Mu_1e28.HLTOverallPrescale = cms.uint32(1)
process.Mu_1e28.throw = False
process.Mu_1e28.andOr = True

process.filterSdMu_1e28 = cms.Path(process.Mu_1e28)


### EGMonitor SD
process.EGMonitor_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.EGMonitor_1e28.HLTPaths = (
"HLT_L1SingleEG2",
"HLT_L1SingleEG5",
"HLT_L1SingleEG8",
"HLT_L1DoubleEG5",
"HLT_EgammaSuperClusterOnly_L1R",
"HLT_L1SingleEG20_NoBPTX",
"HLT_L1SingleEG2_NoBPTX",
"HLT_L1SingleEG5_NoBPTX"
)
process.EGMonitor_1e28.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1,1,1)
process.EGMonitor_1e28.HLTOverallPrescale = cms.uint32(1)
process.EGMonitor_1e28.throw = False
process.EGMonitor_1e28.andOr = True

process.filterSdEGMonitor_1e28 = cms.Path(process.EGMonitor_1e28)



### EG SD
process.EG_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.EG_1e28.HLTPaths = (
"HLT_Photon10_L1R",
"HLT_Photon15_L1R",
"HLT_Photon15_LooseEcalIso_L1R",
"HLT_Photon20_L1R",
"HLT_Photon30_L1R_8E29",
"HLT_DoublePhoton4_Jpsi_L1R",
"HLT_DoublePhoton4_Upsilon_L1R",
"HLT_DoublePhoton4_eeRes_L1R",
"HLT_DoublePhoton5_eeRes_L1R", #added to match the /cdaq/physics/firstCollisions10/v2.0/HLT_7TeV/V5 table
"HLT_DoublePhoton5_Jpsi_L1R",
"HLT_DoublePhoton5_Upsilon_L1R",
"HLT_DoublePhoton5_L1R",
"HLT_DoublePhoton10_L1R",
"HLT_DoubleEle5_SW_L1R",
"HLT_Ele20_LW_L1R",
"HLT_Ele15_SiStrip_L1R",
"HLT_Ele15_SC10_LW_L1R",
"HLT_Ele15_LW_L1R",
"HLT_Ele10_LW_EleId_L1R",
"HLT_Ele10_LW_L1R",
"HLT_Photon15_TrackIso_L1R"
)
process.EG_1e28.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
process.EG_1e28.HLTOverallPrescale = cms.uint32(1)
process.EG_1e28.throw = False
process.EG_1e28.andOr = True

process.filterSdEG_1e28 = cms.Path(process.EG_1e28)




### JetMET AOD CS
process.DiJetAve_1e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.DiJetAve_1e29.HLTPaths = ("HLT_DiJetAve15U_8E29","HLT_DiJetAve30U_8E29")
process.DiJetAve_1e29.HLTPathsPrescales  = cms.vuint32(1,1)
process.DiJetAve_1e29.HLTOverallPrescale = cms.uint32(1)
process.DiJetAve_1e29.andOr = True

process.filterCsDiJetAve_1e29 = cms.Path(process.DiJetAve_1e29)



### Onia skim CS
process.goodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isGlobalMuon || (isTrackerMuon && numberOfMatches('SegmentAndTrackArbitration')>0)"),
)
process.diMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodMuons goodMuons"),
    checkCharge = cms.bool(False),
    cut         = cms.string("mass > 2"),
)
process.diMuonFilter = cms.EDFilter("CandViewCountFilter",
    src       = cms.InputTag("diMuons"),
    minNumber = cms.uint32(1),
)
process.Skim_diMuons = cms.Path(
    process.goodMuons    *
    process.diMuons      *
    process.diMuonFilter
)


### Tau skim CS
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff")
process.load("HLTrigger/HLTfilters/hltLevel1GTSeed_cfi")
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('(0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39))')

process.scrapping = cms.EDFilter("FilterOutScraping",
    	applyfilter = cms.untracked.bool(True),
    	debugOn = cms.untracked.bool(False),
        numtrack = cms.untracked.uint32(10),
        thresh = cms.untracked.double(0.25)
)

process.PFTausSelected = cms.EDFilter("PFTauSelector",
    src = cms.InputTag("shrinkingConePFTauProducer"),
    discriminators = cms.VPSet(
	cms.PSet( discriminator=cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"),
		   selectionCut=cms.double(0.5)
	),
    ),
    cut = cms.string('et > 15. && abs(eta) < 2.5') 
)

process.PFTauSkimmed = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag('PFTausSelected'),
  minNumber = cms.uint32(1)
)


process.tauFilter = cms.Path(
	process.hltLevel1GTSeed *
	process.scrapping *
	process.PFTausSelected *
	process.PFTauSkimmed
)



process.outputSdJetMETTau = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterSdJetMETTau_1e28')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('SD_JetMETTau')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('SD_JetMETTau_1e28.root')
                                          )

process.outputSdJetMETTauMonitor = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterSdJetMETTauMonitor_1e28')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('SD_JetMETTauMonitor')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('SD_JetMETTauMonitor_1e28.root')
                                          )
process.outputSdMuMonitor = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterSdMuMonitor_1e28')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('SD_MuMonitor')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('SD_MuMonitor_1e28.root')
                                          )

process.outputSdMu = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterSdMu_1e28')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('SD_Mu')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('SD_Mu_1e28.root')
                                          )
process.outputSdEGMonitor = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterSdEGMonitor_1e28')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('SD_EGMonitor')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('SD_EGMonitor_1e28.root')
                                          )
process.outputSdEG = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterSdEG_1e28')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('SD_EG')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('SD_EG_1e28.root')
                                          )


process.outputCsDiJet = cms.OutputModule("PoolOutputModule",
                                         dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('USER'),
        filterName = cms.untracked.string('CS_DiJetAve')),
                                         outputCommands = cms.untracked.vstring(
        'drop *',
        #------- CaloJet collections ------
        'keep recoCaloJets_kt4CaloJets_*_*',
        'keep recoCaloJets_kt6CaloJets_*_*',
        'keep recoCaloJets_ak5CaloJets_*_*',
        'keep recoCaloJets_ak7CaloJets_*_*',
        'keep recoCaloJets_iterativeCone5CaloJets_*_*',
        #------- CaloJet ID ---------------
        'keep *_kt4JetID_*_*',
        'keep *_kt6JetID_*_*',
        'keep *_ak5JetID_*_*',
        'keep *_ak7JetID_*_*',
        'keep *_ic5JetID_*_*', 
        #------- PFJet collections ------  
        'keep recoPFJets_kt4PFJets_*_*',
        'keep recoPFJets_kt6PFJets_*_*',
        'keep recoPFJets_ak5PFJets_*_*',
        'keep recoPFJets_ak7PFJets_*_*',
        'keep recoPFJets_iterativeCone5PFJets_*_*',
        #------- JPTJet collections ------
        'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*',
        #'keep *_ak5JPTJets_*_*',
        #'keep *_iterativeCone5JPTJets_*_*',
        #------- Trigger collections ------
        'keep edmTriggerResults_TriggerResults_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep L1GlobalTriggerObjectMapRecord_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        #------- Tracks collection --------
        'keep recoTracks_generalTracks_*_*',
        #------- CaloTower collection -----
        'keep *_towerMaker_*_*',
        #------- Various collections ------
        'keep *_EventAuxilary_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_hcalnoise_*_*',
        #------- MET collections ----------
        'keep *_metHO_*_*',
        'keep *_metNoHF_*_*',
        'keep *_metNoHFHO_*_*', 
        'keep *_met_*_*'),
                                         SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterCsDiJetAve_1e29')), 
                                         fileName = cms.untracked.string('CS_JetAOD_DiJetAve_1e28.root')
                                         )

process.outputCsOnia = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('CS_Onia')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('Skim_diMuons')),                                        
                                        outputCommands = process.RECOEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Onia_1e28.root')
                                        )

process.outputCsTau = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-RECO'),
        filterName = cms.untracked.string('CS_Tau')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('tauFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Tau_1e28.root')
                                        )



process.this_is_the_end = cms.EndPath(
process.outputSdJetMETTau        +
process.outputSdJetMETTauMonitor +
process.outputSdMuMonitor        +
process.outputSdMu               +
process.outputSdEGMonitor        +
process.outputSdEG               +
process.outputCsDiJet            +
process.outputCsOnia             +
process.outputCsTau
)
