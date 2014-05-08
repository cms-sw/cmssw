import FWCore.ParameterSet.Config as cms


process = cms.Process("makeSD")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('JetMETAOD central skim'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/test/CSmaker_JetMETAOD_PDJetMET_35e29_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR10_P_V7::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/FC12B62D-D39E-DF11-A172-001617E30D40.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/F2C6D071-C59E-DF11-97FC-0030487CD184.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/C2ABC2C1-C49E-DF11-BB3B-0030487C90C4.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/B204921D-D89E-DF11-A164-0030487C6062.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/AA4601FB-CE9E-DF11-BF26-0030487A195C.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/A8535E5E-C79E-DF11-9C13-001D09F24493.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/94920AC0-CA9E-DF11-A18C-003048D2C174.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/6CF950BE-DD9E-DF11-A8CA-001D09F23F2A.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/64B83DAB-CF9E-DF11-9BAD-003048F110BE.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/60306759-E39E-DF11-9A9C-0030487A1884.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/427127BE-C89E-DF11-87E6-003048F117EA.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/4205F80E-C39E-DF11-960A-001D09F24303.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/3CDB7561-DC9E-DF11-8517-001D09F24664.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/2C3BE7B5-C69E-DF11-917D-001617C3B5F4.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/28F11C21-D89E-DF11-BD4E-0030487A18D8.root',
        '/store/data/Run2010A/JetMET/RECO/v4/000/142/187/1E464E9C-DB9E-DF11-8FDC-001617DBD556.root'
        )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

import HLTrigger.HLTfilters.hltHighLevelDev_cfi

### JetMET AOD CS
process.DiJetAve_1e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.DiJetAve_1e29.HLTPaths = ("HLT_DiJetAve15U","HLT_DiJetAve30U","HLT_DiJetAve50U","HLT_DiJetAve70U")
process.DiJetAve_1e29.HLTPathsPrescales  = cms.vuint32(1,1,1,1)
process.DiJetAve_1e29.HLTOverallPrescale = cms.uint32(1)
process.DiJetAve_1e29.andOr = True

process.filterCsDiJetAve_1e29 = cms.Path(process.DiJetAve_1e29)




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
        #'keep *_ak4JPTJets_*_*',
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
                                         fileName = cms.untracked.string('CS_JetAOD_DiJetAve_1e29.root')
                                         )



process.this_is_the_end = cms.EndPath(
process.outputCsDiJet
)
