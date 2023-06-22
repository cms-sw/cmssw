import FWCore.ParameterSet.Config as cms

#this is the config to define t&p based DQM offline monitoring for e/gamma

etBinsStd=cms.vdouble(5,10,12.5,15,17.5,20,22.5,25,30,35,40,45,50,60,80,100,150,200,250,300,350,400)
scEtaBinsStd = cms.vdouble(-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5)
phiBinsStd = cms.vdouble(-3.32,-2.97,-2.62,-2.27,-1.92,-1.57,-1.22,-0.87,-0.52,-0.18,0.18,0.52,0.87,1.22,1.57,1.92,2.27,2.62,2.97,3.32)

etRangeCut= cms.PSet(
    rangeVar=cms.string("et"),
    allowedRanges=cms.vstring("0:10000"),
    )
ecalBarrelEtaCut=cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-1.4442:1.4442")
    )
ecalEndcapEtaCut=cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-2.5:-1.556","1.556:2.5")
    )
ecalEndcapHighEtaCut=cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-3.0:-2.5","2.5:3.0")
    )
ecalEndcapPosHighEtaCut= cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("2.5:3.0"),
    )
ecalEndcapNegHighEtaCut= cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-3.0:-2.5"),
    )

ecalBarrelAndEndcapEtaCut = cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-1.4442:1.4442","-2.5:-1.556","1.556:2.5"),
    )
hcalPosEtaCut= cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("1.3:1.4442","1.556:2.5"),
    )

hcalNegEtaCut= cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-2.5:-1.556","-1.4442:-1.3"),
    )

muonEtaCut=cms.PSet(
    rangeVar=cms.string("eta"),
    allowedRanges=cms.vstring("-2.4:2.4")
    )
tagAndProbeConfigEleWPTight = cms.PSet(
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    tagColl = cms.InputTag("gedGsfElectrons"),
    probeColl = cms.InputTag("gedGsfElectrons"),
    tagVIDCuts = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-Summer16-80X-V1-tight"),
    probeVIDCuts = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-Summer16-80X-V1-tight"),
    sampleTrigRequirements = cms.PSet(
        hltInputTag = cms.InputTag("TriggerResults","","HLT"),
        hltPaths = cms.vstring("HLT_Ele30_WPTight_Gsf_v*","HLT_Ele32_WPTight_Gsf_v*","HLT_Ele35_WPTight_Gsf_v*"
                               "HLT_Ele38_WPTight_Gsf_v*",
                               "HLT_Ele32_WPTight_Gsf_L1DoubleEG_v*",
                               )
        ),
    #it is intended that these are the filters of the triggers listed for sampleTrigRequirements
    tagFilters = cms.vstring("hltEle30WPTightGsfTrackIsoFilter",
                             "hltEle32WPTightGsfTrackIsoFilter"
                             "hltEle35noerWPTightGsfTrackIsoFilter"
                             "hltEle38noerWPTightGsfTrackIsoFilter"
                             "hltEle32L1DoubleEGWPTightGsfTrackIsoFilter" ),
    tagFiltersORed = cms.bool(True),
    tagRangeCuts = cms.VPSet(ecalBarrelEtaCut),
    probeFilters = cms.vstring(),
    probeFiltersORed = cms.bool(False),
    probeRangeCuts = cms.VPSet(ecalBarrelAndEndcapEtaCut),
    minTagProbeDR = cms.double(0),
    minMass = cms.double(70.0),
    maxMass = cms.double(110.0),
    requireOpSign = cms.bool(False),
      
) 


tagAndProbeElePhoConfigEleWPTight = tagAndProbeConfigEleWPTight.clone(
    probeColl=cms.InputTag("gedPhotons"),
    probeVIDCuts=cms.InputTag("cutBasedPhotonID-Spring16-V2p2-loose"),
    minTagProbeDR=cms.double(0.1)
)

tagAndProbeElePhoHighEtaConfigEleWPTight = tagAndProbeConfigEleWPTight.clone(
    probeColl=cms.InputTag("gedPhotons"),
    probeVIDCuts=cms.InputTag("cutBasedPhotonID-Spring16-V2p2-loose"),
    probeRangeCuts = cms.VPSet(),
    minTagProbeDR=cms.double(0.1)
)

tagAndProbeMuonEleConfigIsoMu = cms.PSet(
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    tagColl = cms.InputTag("muons"),
    probeColl = cms.InputTag("gedGsfElectrons"),
    tagVIDCuts = cms.InputTag("egmDQMSelectedMuons"),
    probeVIDCuts = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-Summer16-80X-V1-tight"),
    sampleTrigRequirements = cms.PSet(
        hltInputTag = cms.InputTag("TriggerResults","","HLT"),
        hltPaths = cms.vstring("HLT_IsoMu27_v*")
                               
        ),
    #it is intended that these are the filters of the triggers listed for sampleTrigRequirements
    tagFilters = cms.vstring("hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07"),
    tagFiltersORed = cms.bool(True),
    tagRangeCuts = cms.VPSet(muonEtaCut),
    probeFilters = cms.vstring(),
    probeFiltersORed = cms.bool(False),
    probeRangeCuts = cms.VPSet(ecalBarrelAndEndcapEtaCut),
    minTagProbeDR = cms.double(0.4),
    minMass = cms.double(-1),
    maxMass = cms.double(-1),
    requireOpSign = cms.bool(False),
    )


tagAndProbeMuonPhoConfigIsoMu = tagAndProbeMuonEleConfigIsoMu.clone(
    probeColl=cms.InputTag("gedPhotons"),
    probeVIDCuts=cms.InputTag("cutBasedPhotonID-Spring16-V2p2-loose"),
)

egammaStdHistConfigs = cms.VPSet(
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("et"),
        nameSuffex=cms.string("_EBvsEt"),
        rangeCuts=cms.VPSet(ecalBarrelEtaCut),
        binLowEdges=etBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("et"),
        nameSuffex=cms.string("_EEvsEt"),
        rangeCuts=cms.VPSet(ecalEndcapEtaCut),
        binLowEdges=etBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("scEta"),
        nameSuffex=cms.string("_vsSCEta"),
        rangeCuts=cms.VPSet(),
        binLowEdges=scEtaBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("phi"),
        nameSuffex=cms.string("_EBvsPhi"),
        rangeCuts=cms.VPSet(ecalBarrelEtaCut),
        binLowEdges=phiBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("phi"),
        nameSuffex=cms.string("_EEvsPhi"),
        rangeCuts=cms.VPSet(ecalEndcapEtaCut),
        binLowEdges=phiBinsStd,
        ),
    cms.PSet(
        histType=cms.string("2D"),
        xVar=cms.string("scEta"),
        yVar=cms.string("phi"),
        nameSuffex=cms.string("_vsSCEtaPhi"), 
        rangeCuts=cms.VPSet(),
        xBinLowEdges=scEtaBinsStd,
        yBinLowEdges=phiBinsStd,
        ),
    
    )

egammaHighEtaHistConfigs = cms.VPSet(
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("et"),
        nameSuffex=cms.string("_vsEt"),
        rangeCuts=cms.VPSet(),
        binLowEdges=etBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("scEta"),
        nameSuffex=cms.string("_vsSCEta"),
        rangeCuts=cms.VPSet(),
        binLowEdges=cms.vdouble(-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,2.5,2.6,2.7,2.8,2.9,3.0),
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("phi"),
        nameSuffex=cms.string("_vsPhi"),
        rangeCuts=cms.VPSet(),
        binLowEdges=phiBinsStd,
        ),
    cms.PSet(
        histType=cms.string("2D"),
        xVar=cms.string("scEta"),
        yVar=cms.string("phi"),
        nameSuffex=cms.string("_vsSCEtaPhi"), 
        rangeCuts=cms.VPSet(),
        xBinLowEdges=cms.vdouble(-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,2.5,2.6,2.7,2.8,2.9,3.0),
        yBinLowEdges=phiBinsStd,
        ),
    
    )

egammaStdFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoubleEle33_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltEle33CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoubleEle33_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltDiEle33CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle33CaloIdLMWPMS2Filter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon300_NoHE"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("320:99999")),),
        filterName = cms.string("hltEG300erFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoublePhoton70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltEG70HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoublePhoton70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltDiEG70HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG70HEFilter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoublePhoton85"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("90:99999")),),
        filterName = cms.string("hltEG85HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoublePhoton85"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltDiEG85HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG85HEFilter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DiSC30_18_EIso_AND_HE_Mass70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG30EIso15HE30EcalIsoLastFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DiSC30_18_EIso_AND_HE_Mass70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG18EIso15HE30EcalIsoUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30EIso15HE30EcalIsoLastFilter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("26:99999")),),
        filterName = cms.string("hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg1Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("15:99999")),),
        filterName = cms.string("hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele30_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("33:99999")),),
        filterName = cms.string("hltEle30WPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele32_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32WPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele35_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("42:99999")),),
        filterName = cms.string("hltEle35noerWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele38_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("42:99999")),),
        filterName = cms.string("hltEle38noerWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele32_WPTight_Gsf_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32L1DoubleEGWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon33"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG33L1EG26HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon50"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("55:99999")),),
        filterName = cms.string("hltEG50HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),  
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon75"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltEG75HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon90"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("95:99999")),),
        filterName = cms.string("hltEG90HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon120"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("95:99999")),),
         filterName = cms.string("hltEG120HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon150"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("160:99999")),),
        filterName = cms.string("hltEG150HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon175"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("180:99999")),),
        filterName = cms.string("hltEG175HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Photon200"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("220:99999")),),
        filterName = cms.string("hltEG200HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_CaloJet500"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("500:99999")),),
        filterName = cms.string("hltSingleCaloJet500"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_CaloJet550_NoJetID"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("550:99999")),),
        filterName = cms.string("hltSingleCaloJet550"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele28_HighEta_SC20_Mass55"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("32:99999")),),
        filterName = cms.string("hltEle28HighEtaSC20TrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("55:99999")),),
        filterName = cms.string("hltEle50CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""), 
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele115_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("120:99999")),),
        filterName = cms.string("hltEle115CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele135_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("140:99999")),),
        filterName = cms.string("hltEle135CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DiEle27_WPTightCaloOnly_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27L1DoubleEGWPTightEcalIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DiEle27_WPTightCaloOnly_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltDiEle27L1DoubleEGWPTightEcalIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27L1DoubleEGWPTightEcalIsoFilter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoubleEle27_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoubleEle27_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltDiEle27CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoubleEle25_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("28:99999")),),
        filterName = cms.string("hltEle25CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoubleEle25_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("28:99999")),),
        filterName = cms.string("hltDiEle25CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle25CaloIdLMWPMS2Filter"),
        ),  
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
     cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltDiEle27CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        ),
     cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltEle37CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        )
    
     )


  
egammaPhoHighEtaFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele28_HighEta_SC20_Mass55"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("22:99999")),
                              ecalEndcapHighEtaCut
                              ),
        filterName = cms.string("hltEle28HighEtaSC20Mass55Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle28HighEtaSC20TrackIsoFilter"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele28_HighEta_SC20_Mass55"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("22:99999")),
                              ecalEndcapHighEtaCut
                              ),
        filterName = cms.string("hltEle28HighEtaSC20HcalIsoFilterUnseeded"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle28HighEtaSC20TrackIsoFilter"),
        ),
  
) 
egammaPhoFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltEG20CaloIdLV2ClusterShapeL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltTriEG20CaloIdLV2ClusterShapeUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG20CaloIdLV2ClusterShapeL1TripleEGFilter"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltEG20CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltTriEG20CaloIdLV2R9IdVLR9IdUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG20CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 
    #first seeded leg
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG30CaloIdLV2ClusterShapeL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    #second unseeded leg, 10 GeV 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("15:99999")),),
        filterName = cms.string("hltEG10CaloIdLV2ClusterShapeUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2ClusterShapeL1TripleEGFilter"),
        ), 
    #second unseded leg, 30 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltDiEG30CaloIdLV2EtUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2ClusterShapeL1TripleEGFilter"),
        ), 
    #first seeded leg
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG30CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    #second unseeded leg, 10 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("15:99999")),),
        filterName = cms.string("hltEG10CaloIdLV2R9IdVLR9IdUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
                                     
        ), 
    #second unseeded leg, 30 GeV
     cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltDiEG30CaloIdLV2R9IdVLEtUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 
    #first seeded leg
     cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("38:99999")),),
        filterName = cms.string("hltEG35CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    #second unseeded leg, 5 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("10:99999")),),
        filterName = cms.string("hltEG5CaloIdLV2R9IdVLR9IdUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG35CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 
    #second unseeded leg, 35 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("38:99999")),),
        filterName = cms.string("hltDiEG35CaloIdLV2R9IdVLEtUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG35CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 

) 


egammaMuPhoFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Mu12_DoublePhoton20"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltMu12DiEG20HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu12Diphoton20L1f0L2f8QL3Filtered12"),
        ), 
)

egammaMuEleFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Mu12_DoublePhoton20"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltMu12DiEG20HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu12Diphoton20L1f0L2f8QL3Filtered12"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Mu37_Ele27_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu16orMu25L1f0L2f10QL3Filtered37Q"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Mu27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("42:99999")),),
        filterName = cms.string("hltEle37CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu16orMu25L1f0L2f10QL3Filtered27Q"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_DoubleEle33_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltEle33CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele32_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32WPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGM/TagAndProbeEffs/HLT_Ele32_WPTight_Gsf_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32L1DoubleEGWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
egHLTDQMOfflineTnPSource = DQMEDAnalyzer("HLTEleTagAndProbeOfflineSource",
                                          tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeConfigEleWPTight,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("eleWPTightTag_"),
            filterConfigs = egammaStdFiltersToMonitor,
        ),           
        )
)

egHLTElePhoHighEtaDQMOfflineTnPSource = DQMEDAnalyzer("HLTElePhoTagAndProbeOfflineSource",
                                                       tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeElePhoHighEtaConfigEleWPTight,
            histConfigs = egammaHighEtaHistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoHighEtaProbe_"),
            filterConfigs = egammaPhoHighEtaFiltersToMonitor,
        ),           
        )
)
egHLTElePhoDQMOfflineTnPSource = DQMEDAnalyzer("HLTElePhoTagAndProbeOfflineSource",
                                                tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeElePhoConfigEleWPTight,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoProbe_"),
            filterConfigs = egammaPhoFiltersToMonitor,
        ),           
        )
)

egHLTMuonEleDQMOfflineTnPSource = DQMEDAnalyzer("HLTMuEleTagAndProbeOfflineSource",
                                                 tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeMuonEleConfigIsoMu,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("muonIsoMuTagEleProbe_"),
            filterConfigs = egammaMuEleFiltersToMonitor,
        ),           
        )
)
egHLTMuonPhoDQMOfflineTnPSource = DQMEDAnalyzer("HLTMuPhoTagAndProbeOfflineSource",
                                                 tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeMuonPhoConfigIsoMu,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("muonIsoMuTagPhoProbe_"),
            filterConfigs = egammaMuPhoFiltersToMonitor,
        ),           
        )
)


from RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cff import egmGsfElectronIDs

egmGsfElectronIDsForDQM = egmGsfElectronIDs.clone()
egmGsfElectronIDsForDQM.physicsObjectsIDs = cms.VPSet()
egmGsfElectronIDsForDQM.physicsObjectSrc == cms.InputTag('gedGsfElectrons')
#note: be careful here to when selecting new ids that the vid tools dont do extra setup for them
#for example the HEEP cuts need an extra producer which vid tools automatically handles
from PhysicsTools.SelectorUtils.tools.vid_id_tools import setupVIDSelection
my_id_modules = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff']
for id_module_name in my_id_modules: 
    idmod= __import__(id_module_name, globals(), locals(), ['idName','cutFlow'])
    for name in dir(idmod):
        item = getattr(idmod,name)
        if hasattr(item,'idName') and hasattr(item,'cutFlow'):
            setupVIDSelection(egmGsfElectronIDsForDQM,item)


from RecoEgamma.PhotonIdentification.photonIDValueMapProducer_cff import photonIDValueMapProducer
from RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi import egmPhotonIDs
photonIDValueMapProducerForDQM = photonIDValueMapProducer.clone(
    src="gedPhotons",
    vertices="offlinePrimaryVertices",
    ebReducedRecHitCollection="reducedEcalRecHitsEB",
    eeReducedRecHitCollection="reducedEcalRecHitsEE",
    esReducedRecHitCollection="reducedEcalRecHitsES",
    pfCandidates="particleFlow",
    isAOD=True,
)
egmPhotonIDsForDQM = egmPhotonIDs.clone()
#note: be careful here to when selecting new ids that the vid tools dont do extra setup for them
#for example the HEEP cuts need an extra producer which vid tools automatically handles
from PhysicsTools.SelectorUtils.tools.vid_id_tools import setupVIDSelection
my_id_modules = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_RunIIIWinter22_122X_V1_cff']
for id_module_name in my_id_modules: 
    idmod= __import__(id_module_name, globals(), locals(), ['idName','cutFlow'])
    for name in dir(idmod):
        item = getattr(idmod,name)
        if hasattr(item,'idName') and hasattr(item,'cutFlow'):
            setupVIDSelection(egmPhotonIDsForDQM,item)
egmPhotonIDSequenceForDQM = cms.Sequence(photonIDValueMapProducerForDQM*
                                         egmPhotonIDsForDQM)

egmDQMSelectedMuons = cms.EDProducer("HLTDQMMuonSelector",
                                     objs=cms.InputTag("muons"),
                                     vertices=cms.InputTag("offlinePrimaryVertices"),
                                     selection=cms.string("pt > 20"),
                                     muonSelectionType=cms.string("tight")
                                     )
egmMuonIDSequenceForDQM = cms.Sequence(egmDQMSelectedMuons)
