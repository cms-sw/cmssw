import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")


process.MessageLogger = cms.Service(
        "MessageLogger",
            categories = cms.untracked.vstring('info', 'debug','cout')
            )

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)


# source
process.source = cms.Source("PoolSource", 
     #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/r/rompotis/RedigiSummer08RootTrees/WenuRedigi_RECO_SAMPLE.root')
     fileNames = cms.untracked.vstring(
    'file:zee_Summer09-MC_31X_V3_AODSIM_v1_AODSIM.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

## Load additional processes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## global tags:
process.GlobalTag.globaltag = cms.string('MC_31X_V5::All')
#process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All')

process.load("Configuration.StandardSequences.MagneticField_cff")

# this filter produces patElectrons and patCaloMets to be used in the following
process.aod2patFilter = cms.EDFilter('aod2patFilterZee',
                                     electronCollectionTag = cms.untracked.InputTag("gsfElectrons","","RECO"),
                                     metCollectionTag = cms.untracked.InputTag("met","","RECO"),
    )

##############################################################################
##
##  the filter to select the candidates from the data samples
##
##
## WARNING: you may want to modify this item:
HLT_process_name = "HLT8E29"   # options: HLT or HLT8E29
# trigger path selection
HLT_path_name    = "HLT_Ele15_LW_L1R"
# trigger filter name
HLT_filter_name  =  "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter"
#
process.zeeFilter = cms.EDFilter('ZeeCandidateFilter',
                                  # cuts
                                  ETCut = cms.untracked.double(20.),
                                  METCut = cms.untracked.double(0.),
                                  # trigger here
                                  triggerCollectionTag = cms.untracked.InputTag("TriggerResults","",HLT_process_name),
                                  triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",HLT_process_name),
                                  hltpath = cms.untracked.string(HLT_path_name),
                                  hltpathFilter = cms.untracked.InputTag(HLT_filter_name,"",HLT_process_name),
                                  electronMatched2HLT = cms.untracked.bool(False),
                                  electronMatched2HLT_DR = cms.untracked.double(0.2),
                                  # electrons and MET
                                  electronCollectionTag = cms.untracked.InputTag("aod2patFilter","patElectrons","PAT"),
                                  metCollectionTag = cms.untracked.InputTag("aod2patFilter","patCaloMets","PAT")

                                  )
####################################################################################
##
## the W selection that you prefer

selection_a2 = cms.PSet (
    trackIso_EB = cms.untracked.double(7.2),
    ecalIso_EB = cms.untracked.double(5.7),
    hcalIso_EB = cms.untracked.double(8.1),
    sihih_EB = cms.untracked.double(0.01),
    dphi_EB = cms.untracked.double(1000.),
    deta_EB = cms.untracked.double(0.0071),
    hoe_EB = cms.untracked.double(1000),

    trackIso_EE = cms.untracked.double(5.1),
    ecalIso_EE = cms.untracked.double(5.0),
    hcalIso_EE = cms.untracked.double(3.4),
    sihih_EE = cms.untracked.double(0.028),
    dphi_EE = cms.untracked.double(1000.),
    deta_EE = cms.untracked.double(0.0066),
    hoe_EE = cms.untracked.double(1000.)
    )

selection_test = cms.PSet (
    trackIso_EB = cms.untracked.double(10),
    ecalIso_EB = cms.untracked.double(10),
    hcalIso_EB = cms.untracked.double(10),
    sihih_EB = cms.untracked.double(0.1),
    dphi_EB = cms.untracked.double(1),
    deta_EB = cms.untracked.double(1),
    hoe_EB = cms.untracked.double(1),
    
    trackIso_EE = cms.untracked.double(10),
    ecalIso_EE = cms.untracked.double(10),
    hcalIso_EE = cms.untracked.double(10),
    sihih_EE = cms.untracked.double(1),
    dphi_EE = cms.untracked.double(1),
    deta_EE = cms.untracked.double(1),
    hoe_EE = cms.untracked.double(1)
    )

selection_inverse = cms.PSet (
    trackIso_EB_inv = cms.untracked.bool(True),
    trackIso_EE_inv = cms.untracked.bool(True)
    )

####################################################################################
##
## and the plot creator
process.plotter = cms.EDAnalyzer('ZeePlots',
                                 selection_a2,
                                 selection_inverse,
                                 zeeCollectionTag = cms.untracked.InputTag(
    "zeeFilter","selectedZeeCandidates","PAT")
                                 )



process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.aod2patFilter +process.zeeFilter + process.plotter)
#process.p = cms.Path(process.aod2patFilter + process.eca)



#### SET OF Trigger names for AOD - 321
#
#  HLTPath_[0] = "HLT_Ele10_LW_L1R";
#  HLTFilterType_[0] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter","","HLT8E29");
#  HLTPath_[1] = "HLT_Ele10_LW_EleId_L1R";
#  HLTFilterType_[1] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","","HLT8E29");
#  HLTPath_[2] = "HLT_Ele15_LW_L1R";
#  HLTFilterType_[2] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","","HLT8E29");
#  HLTPath_[3] = "HLT_Ele15_SC10_LW_L1R";
#  HLTFilterType_[3] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10","","HLT8E29");
#  HLTPath_[4] = "HLT_Ele15_SiStrip_L1R";
#  HLTFilterType_[4] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter","","HLT8E29");
#  HLTPath_[5] = "HLT_Ele20_LW_L1R";
#  HLTFilterType_[5] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet20","","HLT8E29");
#  HLTPath_[6] = "HLT_DoubleEle5_SW_L1R";
#  HLTFilterType_[6] = edm::InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter","","HLT8E29");
#  HLTPath_[7] = "HLT_Ele15_SC10_LW_L1R";
#  HLTFilterType_[7] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10","","HLT8E29");
#  HLTPath_[8] = "tba";
#  HLTFilterType_[8] = edm::InputTag("tba");
#  HLTPath_[9] = "tba";
#  HLTFilterType_[9] = edm::InputTag("tba");
#  // e31 menu
#  HLTPath_[10] = "HLT_Ele10_SW_L1R";
#  HLTFilterType_[10] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter","","HLT");
#  HLTPath_[11] = "HLT_Ele15_SW_L1R";
#  HLTFilterType_[11] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter","","HLT");
#  HLTPath_[12] = "HLT_Ele15_SiStrip_L1R"; // <--- same as [4]
#  HLTFilterType_[12] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter","","HLT");
#  HLTPath_[13] = "HLT_Ele15_SW_LooseTrackIso_L1R";
#  HLTFilterType_[13] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15LTITrackIsolFilter","","HLT");
#  HLTPath_[14] = "HLT_Ele15_SW_EleId_L1R";
#  HLTFilterType_[14] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter","","HLT");
#  HLTPath_[15] = "HLT_Ele20_SW_L1R";
#  HLTFilterType_[15] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter","","HLT");
#  HLTPath_[16] = "HLT_Ele20_SiStrip_L1R";
#  HLTFilterType_[16] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt20PixelMatchFilter","","HLT");
#  HLTPath_[17] = "HLT_Ele25_SW_L1R";
#  HLTFilterType_[17] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilterESet25","","HLT");
#  HLTPath_[18] = "HLT_Ele25_SW_EleId_LooseTrackIso_L1R";
#  HLTFilterType_[18] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI","","HLT");
#  HLTPath_[19] = "HLT_DoubleEle10_SW_L1R";
#  HLTFilterType_[19] = edm::InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt10PixelMatchFilter","","HLT");
#  HLTPath_[20] = "HLT_Ele15_SC15_SW_EleId_L1R";
#  HLTFilterType_[20] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdESDoubleSC15","","HLT");
#  HLTPath_[21] = "HLT_Ele15_SC15_SW_LooseTrackIso_L1R";
#  HLTFilterType_[21] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15LTIESDoubleSC15","","HLT");
#  HLTPath_[22] = "HLT_Ele20_SC15_SW_L1R";
#  HLTFilterType_[22] = edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt20ESDoubleSC15","","HLT");
