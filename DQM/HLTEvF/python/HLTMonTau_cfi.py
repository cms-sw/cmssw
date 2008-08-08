import FWCore.ParameterSet.Config as cms

hltDoubleTauMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    L2Monitoring = cms.PSet(
        L2AssociationMap = cms.untracked.InputTag("hltL2DoubleTauIsolationProducer","L2TauIsolationInfoAssociator","HLT"),
        doL2Monitoring = cms.untracked.bool(True)
    ),
#    TriggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT"),
    # for data                                      
    TriggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","FU"),
    HistEtMax = cms.untracked.double(100.0),
    outputFile = cms.untracked.string('output.root'),
    prescaleEvt = cms.untracked.int32(-1),
    HistNEtBins = cms.untracked.int32(50),
    MonitorDaemon = cms.untracked.bool(True),
    HistEtMin = cms.untracked.double(0.0),
    MonitorSetup = cms.PSet(
        METCut = cms.untracked.vdouble(0.0),
        L1BackupFilter = cms.untracked.InputTag("N"),
        L2BackupFilter = cms.untracked.InputTag("N"),
        UseBackupTriggers = cms.untracked.bool(False),
        refFilters = cms.untracked.VInputTag(cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter","","FU")),
        Prescales = cms.untracked.vint32(0),
        L25PixelIsolJets = cms.untracked.InputTag("hltIsolatedL25PixelTau","","FU"),
        L25BackupFilter = cms.untracked.InputTag("N"),
        L3BackupFilter = cms.untracked.InputTag("N"),
        NTriggeredTaus = cms.untracked.uint32(2),
        refFilterIDs = cms.untracked.vint32(11),
        L1Seed = cms.untracked.InputTag("hltDoubleTauL1SeedFilter","","FU"),
        MET = cms.untracked.InputTag("hltMet","","FU"),
        L3SiliconIsolJets = cms.untracked.InputTag("NOTHING"),
        refPrescales = cms.untracked.vint32(1),
        refObjectPtCut = cms.untracked.vdouble(15.0),
        refFilterDescriptions = cms.untracked.vstring('DiElectrons'),
        L2EcalIsolJets = cms.untracked.InputTag("hltL2DoubleTauIsolationSelector","Isolated","FU"),
        monitorName = cms.untracked.string('DoubleTau'),
        L2Reco = cms.untracked.InputTag("hltL2DoubleTauJets","","FU"),
        MainFilter = cms.untracked.InputTag("N")
    ),
    DQMFolder = cms.untracked.string('HLT/HLTTAU'),
    L25Monitoring = cms.PSet(
        L25IsolatedTauTagInfo = cms.untracked.InputTag("hltConeIsolationL25PixelTauIsolated","FU"),
        doL25Monitoring = cms.untracked.bool(True)
    ),
    L3Monitoring = cms.PSet(
        L3IsolatedTauTagInfo = cms.untracked.InputTag("NOTHING"),
        doL3Monitoring = cms.untracked.bool(False)
    ),
    HistNEtaBins = cms.untracked.int32(50),
    disable = cms.untracked.bool(False),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)

hltElectronTauMonitor = cms.EDFilter("HLTTauDQMSource",
    L2Monitoring = cms.PSet(
        L2AssociationMap = cms.untracked.InputTag("hltL2ElectronTauIsolationProducer","L2TauIsolationInfoAssociator","FU"),
        doL2Monitoring = cms.untracked.bool(True)
    ),
    TriggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","FU"),
    HistEtMax = cms.untracked.double(100.0),
    outputFile = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(-1),
    HistNEtBins = cms.untracked.int32(50),
    MonitorDaemon = cms.untracked.bool(True),
    HistEtMin = cms.untracked.double(0.0),
    MonitorSetup = cms.PSet(
        METCut = cms.untracked.vdouble(0.0),
        L1BackupFilter = cms.untracked.InputTag("hltFilterIsolatedTauJetsL25ElectronTauNoL1Tau","","FU"),
        L2BackupFilter = cms.untracked.InputTag("hltFilterL25ElectronTauNoL2","","FU"),
        UseBackupTriggers = cms.untracked.bool(True),
        refFilters = cms.untracked.VInputTag(cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter","","FU")),
        L25PixelIsolJets = cms.untracked.InputTag("hltIsolatedL25ElectronTau","","FU"),
        L25BackupFilter = cms.untracked.InputTag("hltFilterL2ElectronTauNoL25","","FU"),
        L3BackupFilter = cms.untracked.InputTag("NOTHING"),
        NTriggeredTaus = cms.untracked.uint32(1),
        refFilterIDs = cms.untracked.vint32(11),
        L1Seed = cms.untracked.InputTag("hltLevel1GTSeedElectronTau","","FU"),
        MET = cms.untracked.InputTag("hltMet"),
        L3SiliconIsolJets = cms.untracked.InputTag("NOTHING"),
        refObjectPtCut = cms.untracked.vdouble(15.0),
        refFilterDescriptions = cms.untracked.vstring('Dielectrons'),
        L2EcalIsolJets = cms.untracked.InputTag("hltL2ElectronTauIsolationSelector","Isolated","FU"),
        monitorName = cms.untracked.string('ElectronTau'),
        L2Reco = cms.untracked.InputTag("hltL2TauJetsElectronTau","","FU"),
        MainFilter = cms.untracked.InputTag("hltFilterIsolatedTauJetsL25ElectronTau","","FU")
    ),
    DQMFolder = cms.untracked.string('HLT/HLTTAU'),
    L25Monitoring = cms.PSet(
        L25IsolatedTauTagInfo = cms.untracked.InputTag("hltConeIsolationL25ElectronTau","","FU"),
        doL25Monitoring = cms.untracked.bool(True)
    ),
    L3Monitoring = cms.PSet(
        L3IsolatedTauTagInfo = cms.untracked.InputTag("NOTHING"),
        doL3Monitoring = cms.untracked.bool(False)
    ),
    HistNEtaBins = cms.untracked.int32(50),
    disable = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)

hltMuonTauMonitor = cms.EDFilter("HLTTauDQMSource",
    L2Monitoring = cms.PSet(
        L2AssociationMap = cms.untracked.InputTag("hltL2MuonTauIsolationProducer","L2TauIsolationInfoAssociator"),
        doL2Monitoring = cms.untracked.bool(True)
    ),
    TriggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","FU"),
    HistEtMax = cms.untracked.double(100.0),
    outputFile = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(-1),
    HistNEtBins = cms.untracked.int32(50),
    MonitorDaemon = cms.untracked.bool(True),
    HistEtMin = cms.untracked.double(0.0),
    MonitorSetup = cms.PSet(
        METCut = cms.untracked.vdouble(0.0),
        L1BackupFilter = cms.untracked.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTauNoL1Tau","","FU"),
        L2BackupFilter = cms.untracked.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTauNoL2","","FU"),
        UseBackupTriggers = cms.untracked.bool(True),
        refFilters = cms.untracked.VInputTag(cms.InputTag("hltemuNonIsoL1IsoTrackIsolFilter")),
        L25PixelIsolJets = cms.untracked.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau","","FU"),
        L25BackupFilter = cms.untracked.InputTag("hltFilterEcalIsolatedTauJetsMuonTauNoL25","","FU"),
        L3BackupFilter = cms.untracked.InputTag("NOTHING"),
        NTriggeredTaus = cms.untracked.uint32(1),
        refFilterIDs = cms.untracked.vint32(11),
        L1Seed = cms.untracked.InputTag("hltLevel1GTSeedMuonTau","","FU"),
        MET = cms.untracked.InputTag("hltMET"),
        L3SiliconIsolJets = cms.untracked.InputTag("NOTHING"),
        refObjectPtCut = cms.untracked.vdouble(15.0),
        refFilterDescriptions = cms.untracked.vstring('ElectronMuon'),
        L2EcalIsolJets = cms.untracked.InputTag("hltL2MuonTauIsolationSelector","Isolated","FU"),
        monitorName = cms.untracked.string('MuonTau'),
        L2Reco = cms.untracked.InputTag("hltL2TauJetsMuonTau","","FU"),
        MainFilter = cms.untracked.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTau","","FU")
    ),
    DQMFolder = cms.untracked.string('HLT/HLTTAU'),
    L25Monitoring = cms.PSet(
        L25IsolatedTauTagInfo = cms.untracked.InputTag("hltPixelTrackConeIsolationMuonTau","","FU"),
        doL25Monitoring = cms.untracked.bool(True)
    ),
    L3Monitoring = cms.PSet(
        L3IsolatedTauTagInfo = cms.untracked.InputTag("NOTHING"),
        doL3Monitoring = cms.untracked.bool(False)
    ),
    HistNEtaBins = cms.untracked.int32(50),
    disable = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)

hltSingleTauMonitor = cms.EDFilter("HLTTauDQMSource",
    L2Monitoring = cms.PSet(
        L2AssociationMap = cms.untracked.InputTag("hltL2SingleTauIsolationProducer","L2TauIsolationInfoAssociator","FU"),
        doL2Monitoring = cms.untracked.bool(True)
    ),
    TriggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","FU"),
    HistEtMax = cms.untracked.double(100.0),
    outputFile = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(-1),
    HistNEtBins = cms.untracked.int32(50),
    MonitorDaemon = cms.untracked.bool(True),
    HistEtMin = cms.untracked.double(0.0),
    MonitorSetup = cms.PSet(
        METCut = cms.untracked.vdouble(65.0),
        L1BackupFilter = cms.untracked.InputTag("N"),
        L2BackupFilter = cms.untracked.InputTag("N"),
        UseBackupTriggers = cms.untracked.bool(False),
        refFilters = cms.untracked.VInputTag(cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter","","FU")),
        L25PixelIsolJets = cms.untracked.InputTag("hltIsolatedL25SingleTau","","FU"),
        L25BackupFilter = cms.untracked.InputTag("N"),
        L3BackupFilter = cms.untracked.InputTag("N"),
        NTriggeredTaus = cms.untracked.uint32(1),
        refFilterIDs = cms.untracked.vint32(11),
        L1Seed = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","FU"),
        MET = cms.untracked.InputTag("hltMet","","FU"),
        L3SiliconIsolJets = cms.untracked.InputTag("hltIsolatedL3SingleTau","","FU"),
        refObjectPtCut = cms.untracked.vdouble(30.0),
        refFilterDescriptions = cms.untracked.vstring('SingleElectron'),
        L2EcalIsolJets = cms.untracked.InputTag("hltL2SingleTauIsolationSelector","Isolated","FU"),
        monitorName = cms.untracked.string('SingleTau'),
        L2Reco = cms.untracked.InputTag("hltL2SingleTauJets","","FU"),
        MainFilter = cms.untracked.InputTag("N")
    ),
    DQMFolder = cms.untracked.string('HLT/HLTTAU'),
    L25Monitoring = cms.PSet(
        L25IsolatedTauTagInfo = cms.untracked.InputTag("hltConeIsolationL25SingleTau","FU"),
        doL25Monitoring = cms.untracked.bool(True)
    ),
    L3Monitoring = cms.PSet(
        L3IsolatedTauTagInfo = cms.untracked.InputTag("hltConeIsolationL3SingleTau","FU"),
        doL3Monitoring = cms.untracked.bool(True)
    ),
    HistNEtaBins = cms.untracked.int32(50),
    disable = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)

hltSingleTauMETMonitor = cms.EDFilter("HLTTauDQMSource",
    L2Monitoring = cms.PSet(
        L2AssociationMap = cms.untracked.InputTag("hltL2SingleTauMETIsolationProducer","L2TauIsolationInfoAssociator","FU"),
        doL2Monitoring = cms.untracked.bool(True)
    ),
    TriggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","FU"),
    HistEtMax = cms.untracked.double(100.0),
    outputFile = cms.untracked.string('DQMOutput.root'),
    prescaleEvt = cms.untracked.int32(-1),
    HistNEtBins = cms.untracked.int32(50),
    MonitorDaemon = cms.untracked.bool(True),
    HistEtMin = cms.untracked.double(0.0),
    MonitorSetup = cms.PSet(
        METCut = cms.untracked.vdouble(40.0),
        L1BackupFilter = cms.untracked.InputTag("N"),
        L2BackupFilter = cms.untracked.InputTag("N"),
        UseBackupTriggers = cms.untracked.bool(False),
        refFilters = cms.untracked.VInputTag(cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter","","FU")),
        L25PixelIsolJets = cms.untracked.InputTag("hltIsolatedL25SingleTauMET","FU"),
        L25BackupFilter = cms.untracked.InputTag("N"),
        L3BackupFilter = cms.untracked.InputTag("N"),
        NTriggeredTaus = cms.untracked.uint32(1),
        refFilterIDs = cms.untracked.vint32(11),
        L1Seed = cms.untracked.InputTag("hltSingleTauMETL1SeedFilter","","FU"),
        MET = cms.untracked.InputTag("hltMet","","FU"),
        L3SiliconIsolJets = cms.untracked.InputTag("hltIsolatedL3SingleTauMET","FU"),
        refObjectPtCut = cms.untracked.vdouble(20.0),
        refFilterDescriptions = cms.untracked.vstring('SingleElectron'),
        L2EcalIsolJets = cms.untracked.InputTag("hltL2SingleTauMETIsolationSelector","Isolated","FU"),
        monitorName = cms.untracked.string('SingleTauMET'),
        L2Reco = cms.untracked.InputTag("hltL2SingleTauMETJets","","FU"),
        MainFilter = cms.untracked.InputTag("N")
    ),
    DQMFolder = cms.untracked.string('HLT/HLTTAU'),
    L25Monitoring = cms.PSet(
        L25IsolatedTauTagInfo = cms.untracked.InputTag("hltConeIsolationL25SingleTauMET","FU"),
        doL25Monitoring = cms.untracked.bool(True)
    ),
    L3Monitoring = cms.PSet(
        L3IsolatedTauTagInfo = cms.untracked.InputTag("hltConeIsolationL3SingleTauMET","FU"),
        doL3Monitoring = cms.untracked.bool(True)
    ),
    HistNEtaBins = cms.untracked.int32(50),
    disable = cms.untracked.bool(False),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)

hltMonTauReco = cms.Sequence(hltElectronTauMonitor+hltMuonTauMonitor+hltDoubleTauMonitor+hltSingleTauMonitor+hltSingleTauMETMonitor)

