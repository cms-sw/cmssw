import FWCore.ParameterSet.Config as cms


process = cms.Process("minBiasSkim")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Skim for MinBias'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Skimming/test/SkimMinBias_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR10_P_V7::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/EE887979-5CA3-DF11-94C2-0019DB29C614.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/D2D5DCFA-64A3-DF11-92F4-0030487C7392.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/D028C537-58A3-DF11-A4F1-0019B9F709A4.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/C4D9CBA1-60A3-DF11-B1CD-0030487CD7CA.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/907930C5-62A3-DF11-A452-003048D3756A.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/7E67D6FA-64A3-DF11-8E8F-0030487CD16E.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/7E5B8CB9-5BA3-DF11-BFC1-001617C3B76A.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/7C38E794-5EA3-DF11-9741-001D09F2532F.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/7627FE21-77A3-DF11-8253-001D09F231C9.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/64E5C041-66A3-DF11-A830-0030487C7828.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/5C3876E0-5DA3-DF11-B5F2-001D09F290CE.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/3C9B7F9D-59A3-DF11-93D1-0019DB29C614.root',
        '/store/data/Run2010A/MinimumBias/RAW/v1/000/142/558/24F23D4F-5AA3-DF11-A0BB-001D09F24682.root'
        )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100


import HLTrigger.HLTfilters.triggerResultsFilter_cfi
process.minBiasSkim = HLTrigger.HLTfilters.triggerResultsFilter_cfi.triggerResultsFilter.clone()
process.minBiasSkim.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
process.minBiasSkim.l1tResults = cms.InputTag('')
process.minBiasSkim.throw      = cms.bool(False)
process.minBiasSkim.triggerConditions = cms.vstring(
#JetMET PD
    'HLT_HT100U', 
    'HLT_MET100', 
    'HLT_MET45', 
    'HLT_QuadJet15U', 
    'HLT_DiJetAve30U', 
    'HLT_DiJetAve15U', 
    'HLT_FwdJet20U', 
    'HLT_Jet50U', 
    'HLT_Jet30U', 
    'HLT_Jet15U', 
    'HLT_DoubleJet15U_ForwardBackward', 
    'HLT_Jet15U_HcalNoiseFiltered', 
    'HLT_DiJetAve50U', 
    'HLT_Jet100U', 
    'HLT_Jet70U', 
    'HLT_EcalOnly_SumEt160', 
    'HLT_L1ETT100',
#EG PD
    'HLT_DoublePhoton5_Upsilon_L1R', 
    'HLT_DoublePhoton5_Jpsi_L1R', 
    'HLT_Photon20_Cleaned_L1R', 
    'HLT_DoubleEle10_SW_L1R', 
    'HLT_DoublePhoton15_L1R', 
    'HLT_Ele15_SW_EleId_L1R', 
    'HLT_Ele15_SW_L1R', 
    'HLT_Ele20_SW_L1R', 
    'HLT_DoublePhoton5_CEP_L1R', 
    'HLT_Ele10_SW_EleId_L1R', 
    'HLT_Photon30_Cleaned_L1R', 
    'HLT_Photon50_L1R', 
    'HLT_DoubleEle4_SW_eeRes_L1R', 
    'HLT_Ele15_SW_CaloEleId_L1R', 
    'HLT_Photon50_Cleaned_L1R', 
    'HLT_DoublePhoton20_L1R', 
    'HLT_Ele25_SW_L1R',
#EGMonitor PD
    'HLT_L1SingleEG2', 
    'HLT_DoublePhoton10_L1R', 
    'HLT_Photon10_Cleaned_L1R', 
    'HLT_L1DoubleEG5', 
    'HLT_Ele15_SiStrip_L1R', 
    'HLT_Ele15_LW_L1R', 
    'HLT_L1SingleEG8', 
    'HLT_L1SingleEG5', 
    'HLT_SelectEcalSpikes_L1R', 
    'HLT_SelectEcalSpikesHighEt_L1R', 
    'HLT_DoublePhoton5_L1R', 
    'HLT_Photon15_Cleaned_L1R', 
    'HLT_Activity_Ecal_SC7', 
    'HLT_Activity_Ecal_SC17', 
    'HLT_Ele20_SiStrip_L1R',
#JetMETTauMonitor PD
    'HLT_L1Jet10U_NoBPTX', 
    'HLT_L1Jet6U', 
    'HLT_L1Jet6U_NoBPTX', 
    'HLT_L1SingleCenJet', 
    'HLT_L1SingleForJet', 
    'HLT_L1SingleTauJet', 
    'HLT_L1MET20', 
    'HLT_L1Jet10U',
#Mu PD
    'HLT_L1Mu14_L1ETM30', 
    'HLT_L1Mu14_L1SingleJet6U', 
    'HLT_L1Mu14_L1SingleEG10', 
    'HLT_L1Mu20', 
    'HLT_DoubleMu3', 
    'HLT_Mu3', 
    'HLT_Mu5', 
    'HLT_Mu9', 
    'HLT_L2Mu9', 
    'HLT_L2Mu11', 
    'HLT_L1Mu30', 
    'HLT_Mu7', 
    'HLT_L2Mu15',
#MuOnia
    'HLT_Mu3_Track0_Jpsi', 
    'HLT_Mu5_Track0_Jpsi', 
    'HLT_Mu3_L2Mu0', 
    'HLT_Mu5_L2Mu0', 
    'HLT_L2DoubleMu0', 
    'HLT_DoubleMu0', 
    'HLT_L1DoubleMuOpen_Tight', 
    'HLT_Mu0_TkMu0_Jpsi', 
    'HLT_Mu0_TkMu0_Jpsi_NoCharge', 
    'HLT_Mu3_TkMu0_Jpsi', 
    'HLT_Mu3_TkMu0_Jpsi_NoCharge', 
    'HLT_Mu5_TkMu0_Jpsi', 
    'HLT_Mu5_TkMu0_Jpsi_NoCharge',
#Commissioning PD
    'HLT_Activity_DT', 
    'HLT_Activity_DT_Tuned', 
    'HLT_Activity_CSC', 
    'HLT_L1_BptxXOR_BscMinBiasOR',
#BTau PD
    'HLT_BTagIP_Jet50U', 
    'HLT_DoubleLooseIsoTau15', 
    'HLT_SingleLooseIsoTau20', 
    'HLT_BTagMu_Jet10U', 
    'HLT_SingleIsoTau20_Trk5', 
    'HLT_SingleLooseIsoTau25_Trk5', 
    'HLT_SingleIsoTau30_Trk5'
)
process.filterCsPdwgMinBias = cms.Path(process.minBiasSkim)



process.skimmedMinBias = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterCsPdwgMinBias')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW'),
        filterName = cms.untracked.string('SkimmedMinBias')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('CS_pdwg_MinBias.root')
                                          )



process.this_is_the_end = cms.EndPath(
process.skimmedMinBias
)
