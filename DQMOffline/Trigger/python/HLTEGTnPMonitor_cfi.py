import FWCore.ParameterSet.Config as cms

#this is the config to define t&p based DQM offline monitoring for e/gamma

etBinsStd=cms.vdouble(5,10,12.5,15,17.5,20,22.5,25,30,35,40,45,50,60,80,100,150,200,250,300,350,400)
scEtaBinsStd = cms.vdouble(-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5)
scEtaBinsHEP17 = cms.vdouble(1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5)
scEtaBinsHEM17 = cms.vdouble(-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3)
phiBinsStd = cms.vdouble(-3.32,-2.97,-2.62,-2.27,-1.92,-1.57,-1.22,-0.87,-0.52,-0.18,0.18,0.52,0.87,1.22,1.57,1.92,2.27,2.62,2.97,3.32)
phiBinsHE17 = cms.vdouble(-0.87,-0.80,-0.73,-0.66,-0.59,-0.52)

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
hcalPhi17Cut = cms.PSet(
    rangeVar=cms.string("phi"),
    allowedRanges=cms.vstring("-0.87:-0.52"),
    )

muonEtaCut=cms.PSet(
    rangeVar=cms.string("eta"),
    allowedRanges=cms.vstring("-2.4:2.4")
    )
tagAndProbeConfigEle27WPTight = cms.PSet(
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    tagColl = cms.InputTag("gedGsfElectrons"),
    probeColl = cms.InputTag("gedGsfElectrons"),
    tagVIDCuts = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-Summer16-80X-V1-tight"),
    probeVIDCuts = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-Summer16-80X-V1-tight"),
    sampleTrigRequirements = cms.PSet(
        hltInputTag = cms.InputTag("TriggerResults","","HLT"),
        hltPaths = cms.vstring("HLT_Ele27_WPTight_Gsf_v*","HLT_Ele32_WPTight_Gsf_v*","HLT_Ele35_WPTight_Gsf_v*"
                               "HLT_Ele38_WPTight_Gsf_v*",
                               "HLT_Ele27_WPTight_Gsf_L1DoubleEG_v*","HLT_Ele32_WPTight_Gsf_L1DoubleEG_v*",
                               )
        ),
    #it is intended that these are the filters of the triggers listed for sampleTrigRequirements
    tagFilters = cms.vstring("hltEle27WPTightGsfTrackIsoFilter",
                             "hltEle32WPTightGsfTrackIsoFilter"
                             "hltEle35noerWPTightGsfTrackIsoFilter"
                             "hltEle38noerWPTightGsfTrackIsoFilter"
                             "hltEle27L1DoubleEGWPTightGsfTrackIsoFilter",
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
tagAndProbeConfigEle27WPTightHEP17 = tagAndProbeConfigEle27WPTight.clone( 
    probeRangeCuts = cms.VPSet(
        hcalPosEtaCut,
        hcalPhi17Cut,
))
tagAndProbeConfigEle27WPTightHEM17 = tagAndProbeConfigEle27WPTight.clone( 
    probeRangeCuts = cms.VPSet(
        hcalNegEtaCut,
        hcalPhi17Cut,
))


tagAndProbeElePhoConfigEle27WPTight = tagAndProbeConfigEle27WPTight.clone(
    probeColl=cms.InputTag("gedPhotons"),
    probeVIDCuts=cms.InputTag("cutBasedPhotonID-Spring16-V2p2-loose"),
    minTagProbeDR=cms.double(0.1)
)

tagAndProbeElePhoConfigEle27WPTightHEP17 = tagAndProbeElePhoConfigEle27WPTight.clone(
     probeRangeCuts = cms.VPSet(
        hcalPosEtaCut,
        hcalPhi17Cut,
))
tagAndProbeElePhoConfigEle27WPTightHEM17 = tagAndProbeElePhoConfigEle27WPTight.clone(
     probeRangeCuts = cms.VPSet(
        hcalNegEtaCut,
        hcalPhi17Cut,
))

tagAndProbeElePhoHighEtaConfigEle27WPTight = tagAndProbeConfigEle27WPTight.clone(
    probeColl=cms.InputTag("gedPhotons"),
    probeVIDCuts=cms.InputTag("cutBasedPhotonID-Spring16-V2p2-loose"),
    probeRangeCuts = cms.VPSet(),
    minTagProbeDR=cms.double(0.1)
)
tagAndProbeElePhoHighEtaConfigEle27WPTightHEP17 = tagAndProbeElePhoHighEtaConfigEle27WPTight.clone(
     probeRangeCuts = cms.VPSet(
        ecalEndcapPosHighEtaCut,
        hcalPhi17Cut,
))
tagAndProbeElePhoHighEtaConfigEle27WPTightHEM17 = tagAndProbeElePhoHighEtaConfigEle27WPTight.clone(
     probeRangeCuts = cms.VPSet(
        ecalEndcapPosHighEtaCut,
        hcalPhi17Cut,
))


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

tagAndProbeMuonEleConfigIsoMuHEP17 = tagAndProbeMuonEleConfigIsoMu.clone(
    probeRangeCuts = cms.VPSet(
        hcalPosEtaCut,
        hcalPhi17Cut,
        )
    )
tagAndProbeMuonEleConfigIsoMuHEM17 = tagAndProbeMuonEleConfigIsoMu.clone(
    probeRangeCuts = cms.VPSet(
        hcalNegEtaCut,
        hcalPhi17Cut,
        )
    )

tagAndProbeMuonPhoConfigIsoMu = tagAndProbeMuonEleConfigIsoMu.clone(
    probeColl=cms.InputTag("gedPhotons"),
    probeVIDCuts=cms.InputTag("cutBasedPhotonID-Spring16-V2p2-loose"),
)
tagAndProbeMuonPhoConfigIsoMuHEP17 = tagAndProbeMuonPhoConfigIsoMu.clone(
    probeRangeCuts = cms.VPSet(
        hcalPosEtaCut,
        hcalPhi17Cut,
        )
    )
tagAndProbeMuonPhoConfigIsoMuHEM17 = tagAndProbeMuonPhoConfigIsoMu.clone(
    probeRangeCuts = cms.VPSet(
        hcalNegEtaCut,
        hcalPhi17Cut,
        )
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

egammaHEP17HistConfigs = cms.VPSet(
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
        binLowEdges=scEtaBinsHEP17,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("phi"),
        nameSuffex=cms.string("_vsPhi"),
        rangeCuts=cms.VPSet(),
        binLowEdges=phiBinsHE17,
    )
)
egammaHEM17HistConfigs = cms.VPSet(
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
        binLowEdges=scEtaBinsHEM17,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("phi"),
        nameSuffex=cms.string("_vsPhi"),
        rangeCuts=cms.VPSet(),
        binLowEdges=phiBinsHE17,
    )
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
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoubleEle33_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltEle33CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoubleEle33_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltDiEle33CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle33CaloIdLMWPMS2Filter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon300_NoHE"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("320:99999")),),
        filterName = cms.string("hltEG300erFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoublePhoton70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltEG70HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoublePhoton70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltDiEG70HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG70HEFilter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoublePhoton85"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("90:99999")),),
        filterName = cms.string("hltEG85HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoublePhoton85"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltDiEG85HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG85HEFilter"),
        ),
    
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DiSC30_18_EIso_AND_HE_Mass70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG30EIso15HE30EcalIsoLastFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DiSC30_18_EIso_AND_HE_Mass70"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG18EIso15HE30EcalIsoUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30EIso15HE30EcalIsoLastFilter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("26:99999")),),
        filterName = cms.string("hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg1Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("15:99999")),),
        filterName = cms.string("hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele27_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27WPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele32_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32WPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele35_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("42:99999")),),
        filterName = cms.string("hltEle35noerWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele38_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("42:99999")),),
        filterName = cms.string("hltEle38noerWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele27_WPTight_Gsf_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27L1DoubleEGWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele32_WPTight_Gsf_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32L1DoubleEGWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon25"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("28:99999")),),
        filterName = cms.string("hltEG25L1EG18HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),

    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon33"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG33L1EG26HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon50"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("55:99999")),),
        filterName = cms.string("hltEG50HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),  
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon75"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("80:99999")),),
        filterName = cms.string("hltEG75HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon90"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("95:99999")),),
        filterName = cms.string("hltEG90HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon120"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("95:99999")),),
         filterName = cms.string("hltEG120HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon150"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("160:99999")),),
        filterName = cms.string("hltEG150HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon175"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("180:99999")),),
        filterName = cms.string("hltEG175HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Photon200"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("220:99999")),),
        filterName = cms.string("hltEG200HEFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_CaloJet500"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("500:99999")),),
        filterName = cms.string("hltSingleCaloJet500"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_CaloJet550"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("550:99999")),),
        filterName = cms.string("hltSingleCaloJet550"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele28_HighEta_SC20_Mass55"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("32:99999")),),
        filterName = cms.string("hltEle28HighEtaSC20TrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("55:99999")),),
        filterName = cms.string("hltEle50CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""), 
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele115_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("120:99999")),),
        filterName = cms.string("hltEle115CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele135_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("140:99999")),),
        filterName = cms.string("hltEle135CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele145_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("150:99999")),),
        filterName = cms.string("hltEle145CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele200_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("210:99999")),),
        filterName = cms.string("hltEle200CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele250_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("260:99999")),),
        filterName = cms.string("hltEle250CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele300_CaloIdVT_GsfTrkIdT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("310:99999")),),
        filterName = cms.string("hltEle300CaloIdVTGsfTrkIdTGsfDphiFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele20_WPLoose_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltEle20WPLoose1GsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele20_eta2p1_WPLoose_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltEle20erWPLoose1GsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele20_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltEle20WPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DiEle27_WPTightCaloOnly_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27L1DoubleEGWPTightEcalIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DiEle27_WPTightCaloOnly_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltDiEle27L1DoubleEGWPTightEcalIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27L1DoubleEGWPTightEcalIsoFilter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoubleEle27_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoubleEle27_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltDiEle27CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoubleEle25_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("28:99999")),),
        filterName = cms.string("hltEle25CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoubleEle25_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("28:99999")),),
        filterName = cms.string("hltDiEle25CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle25CaloIdLMWPMS2Filter"),
        ),  
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
     cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltDiEle27CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        ),
     cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltEle37CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle27CaloIdLMWPMS2Filter"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele35_WPTight_Gsf_L1EGMT"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("38:99999")),),
        filterName = cms.string("hltSingleEle35WPTightGsfL1EGMTTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    
     )


  
egammaPhoHighEtaFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele28_HighEta_SC20_Mass55"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("22:99999")),
                              ecalEndcapHighEtaCut
                              ),
        filterName = cms.string("hltEle28HighEtaSC20Mass55Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEle28HighEtaSC20TrackIsoFilter"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele28_HighEta_SC20_Mass55"),
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
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltEG20CaloIdLV2ClusterShapeL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltTriEG20CaloIdLV2ClusterShapeUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG20CaloIdLV2ClusterShapeL1TripleEGFilter"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltEG20CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_20_20_20_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltTriEG20CaloIdLV2R9IdVLR9IdUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG20CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 
    #first seeded leg
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG30CaloIdLV2ClusterShapeL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    #second unseeded leg, 10 GeV 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("15:99999")),),
        filterName = cms.string("hltEG10CaloIdLV2ClusterShapeUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2ClusterShapeL1TripleEGFilter"),
        ), 
    #second unseded leg, 30 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltDiEG30CaloIdLV2EtUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2ClusterShapeL1TripleEGFilter"),
        ), 
    #first seeded leg
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEG30CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    #second unseeded leg, 10 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("15:99999")),),
        filterName = cms.string("hltEG10CaloIdLV2R9IdVLR9IdUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
                                     
        ), 
    #second unseeded leg, 30 GeV
     cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_30_30_10_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltDiEG30CaloIdLV2R9IdVLEtUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG30CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 
    #first seeded leg
     cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("38:99999")),),
        filterName = cms.string("hltEG35CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ), 
    #second unseeded leg, 5 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("10:99999")),),
        filterName = cms.string("hltEG5CaloIdLV2R9IdVLR9IdUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG35CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 
    #second unseeded leg, 35 GeV
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_TriplePhoton_35_35_5_CaloIdLV2_R9IdVL"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("38:99999")),),
        filterName = cms.string("hltDiEG35CaloIdLV2R9IdVLEtUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltEG35CaloIdLV2R9IdVLR9IdL1TripleEGFilter"),
        ), 

) 


egammaMuPhoFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Mu12_DoublePhoton20"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltMu12DiEG20HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu12Diphoton20L1f0L2f8QL3Filtered12"),
        ), 
)

egammaMuEleFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Mu12_DoublePhoton20"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("25:99999")),),
        filterName = cms.string("hltMu12DiEG20HEUnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu12Diphoton20L1f0L2f8QL3Filtered12"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Mu37_Ele27_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("30:99999")),),
        filterName = cms.string("hltEle27CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu16orMu25L1f0L2f10QL3Filtered37Q"),
        ), 
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Mu27_Ele37_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("42:99999")),),
        filterName = cms.string("hltEle37CaloIdLMWPMS2UnseededFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string("hltL3fL1sMu16orMu25L1f0L2f10QL3Filtered27Q"),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_DoubleEle33_CaloIdL_MW"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("40:99999")),),
        filterName = cms.string("hltEle33CaloIdLMWPMS2Filter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele32_WPTight_Gsf"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32WPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
    cms.PSet(
        folderName = cms.string("HLT/EGTagAndProbeEffs/HLT_Ele32_WPTight_Gsf_L1DoubleEG"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges=cms.vstring("35:99999")),),
        filterName = cms.string("hltEle32L1DoubleEGWPTightGsfTrackIsoFilter"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
)

egHLTDQMOfflineTnPSource = cms.EDAnalyzer("HLTEleTagAndProbeOfflineSource",
                                          tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeConfigEle27WPTight,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("eleWPTightTag_"),
            filterConfigs = egammaStdFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeConfigEle27WPTightHEM17,
            histConfigs = egammaHEM17HistConfigs,
            baseHistName = cms.string("eleWPTightTag-HEM17_"),
            filterConfigs = egammaStdFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeConfigEle27WPTightHEP17,
            histConfigs = egammaHEP17HistConfigs,
            baseHistName = cms.string("eleWPTightTag-HEP17_"),
            filterConfigs = egammaStdFiltersToMonitor,
        ),
           
        )
)

egHLTElePhoHighEtaDQMOfflineTnPSource = cms.EDAnalyzer("HLTElePhoTagAndProbeOfflineSource",
                                                       tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeElePhoHighEtaConfigEle27WPTight,
            histConfigs = egammaHighEtaHistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoHighEtaProbe_"),
            filterConfigs = egammaPhoHighEtaFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeElePhoHighEtaConfigEle27WPTightHEM17,
            histConfigs = egammaHighEtaHistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoHighEtaProbe-HEM17_"),
            filterConfigs = egammaPhoHighEtaFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeElePhoHighEtaConfigEle27WPTightHEP17,
            histConfigs = egammaHighEtaHistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoHighEtaProbe-HEP17_"),
            filterConfigs = egammaPhoHighEtaFiltersToMonitor,
        ),
           
        )
                                                )
egHLTElePhoDQMOfflineTnPSource = cms.EDAnalyzer("HLTElePhoTagAndProbeOfflineSource",
                                                tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeElePhoConfigEle27WPTight,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoProbe_"),
            filterConfigs = egammaPhoFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeElePhoConfigEle27WPTightHEM17,
            histConfigs = egammaHEM17HistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoProbe-HEM17_"),
            filterConfigs = egammaPhoFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeElePhoConfigEle27WPTightHEP17,
            histConfigs = egammaHEM17HistConfigs,
            baseHistName = cms.string("eleWPTightTagPhoProbe-HEP17_"),
            filterConfigs = egammaPhoFiltersToMonitor,
        ),
           
        )
)

egHLTMuonEleDQMOfflineTnPSource = cms.EDAnalyzer("HLTMuEleTagAndProbeOfflineSource",
                                                 tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeMuonEleConfigIsoMu,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("muonIsoMuTagEleProbe_"),
            filterConfigs = egammaMuEleFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeMuonEleConfigIsoMuHEM17,
            histConfigs = egammaHEM17HistConfigs,
            baseHistName = cms.string("muonIsoMuTagEleProbe-HEM17_"),
            filterConfigs = egammaMuEleFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeMuonPhoConfigIsoMuHEP17,
            histConfigs = egammaHEP17HistConfigs,
            baseHistName = cms.string("muonIsoMuTagEleProbe-HEP17_"),
            filterConfigs = egammaMuEleFiltersToMonitor,
        ),
           
        )
)
egHLTMuonPhoDQMOfflineTnPSource = cms.EDAnalyzer("HLTMuPhoTagAndProbeOfflineSource",
                                                 tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeMuonPhoConfigIsoMu,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("muonIsoMuTagPhoProbe_"),
            filterConfigs = egammaMuPhoFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeMuonPhoConfigIsoMuHEM17,
            histConfigs = egammaHEM17HistConfigs,
            baseHistName = cms.string("muonIsoMuTagPhoProbe-HEM17_"),
            filterConfigs = egammaMuPhoFiltersToMonitor,
        ),
        cms.PSet(
            tagAndProbeMuonPhoConfigIsoMuHEP17,
            histConfigs = egammaHEP17HistConfigs,
            baseHistName = cms.string("muonIsoMuTagPhoProbe-HEP17_"),
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
my_id_modules = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff']
for id_module_name in my_id_modules: 
    idmod= __import__(id_module_name, globals(), locals(), ['idName','cutFlow'])
    for name in dir(idmod):
        item = getattr(idmod,name)
        if hasattr(item,'idName') and hasattr(item,'cutFlow'):
            setupVIDSelection(egmGsfElectronIDsForDQM,item)


from RecoEgamma.PhotonIdentification.PhotonIDValueMapProducer_cfi import photonIDValueMapProducer
from RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi import egmPhotonIDs
egmPhotonIDsForDQM = egmPhotonIDs.clone()
egmPhotonIDsForDQM.physicsObjectsIDs = cms.VPSet()
egmPhotonIDsForDQM.physicsObjectSrc == cms.InputTag('gedPhotons')
#note: be careful here to when selecting new ids that the vid tools dont do extra setup for them
#for example the HEEP cuts need an extra producer which vid tools automatically handles
from PhysicsTools.SelectorUtils.tools.vid_id_tools import setupVIDSelection
my_id_modules = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring16_V2p2_cff']
for id_module_name in my_id_modules: 
    idmod= __import__(id_module_name, globals(), locals(), ['idName','cutFlow'])
    for name in dir(idmod):
        item = getattr(idmod,name)
        if hasattr(item,'idName') and hasattr(item,'cutFlow'):
            setupVIDSelection(egmPhotonIDsForDQM,item)
egmPhotonIDSequenceForDQM = cms.Sequence(photonIDValueMapProducer*
                                         egmPhotonIDsForDQM)

egmDQMSelectedMuons = cms.EDProducer("HLTDQMMuonSelector",
                                     objs=cms.InputTag("muons"),
                                     vertices=cms.InputTag("offlinePrimaryVertices"),
                                     selection=cms.string("pt > 20"),
                                     muonSelectionType=cms.string("tight")
                                     )
egmMuonIDSequenceForDQM = cms.Sequence(egmDQMSelectedMuons)
