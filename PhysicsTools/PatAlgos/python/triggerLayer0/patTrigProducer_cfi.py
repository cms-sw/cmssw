import FWCore.ParameterSet.Config as cms

# Examples for configurations of the trigger primitive producers.
#
# A list of valid HLT path names for CMSSW releases is found in: 
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideGlobalHLT
# Choose the path 'myPath' of your interest. Then you can find the filter modules by doing
# > grep -n "path [myPath]" /afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/[CMSSW version]/src/HLTrigger/Configuration/data/HLT_{2E30,1E32}.cff
# 
# BEWARE: Use the list of the CMSSW version the data has been  p r o d u c e d  with,
#         not of that you are running with!
#         Use this version also in the 'grep' command -- not $CMSSW_VERSION!
# BEWARE: Only those parts of the paths starting with  l o w e r  case letters are modules!
#         Capital letters indicate sequences, which you also can 'grep' for then.
# BEWARE: Make sure to choose filter modules only, not e.g. prescaler modules!
# modules to produce pat::TriggerPrimitiveCollections,
# grouped by POG (corresponding to trigger config files)
# The module names without "pat" indicate the HLT path name.

# Egamma triggers
patCandHLT1ElectronStartup = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter","","HLT")
)

patHLT1Photon = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter","","HLT")
)

patHLT1PhotonRelaxed = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter","","HLT")
)

patHLT2Photon = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1IsoDoublePhotonDoubleEtFilter","","HLT")
)

patHLT2PhotonRelaxed = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter","","HLT")
)

patHLT1Electron = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter","","HLT")
)

patHLT1ElectronRelaxed = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter","","HLT")
)

patHLT2Electron = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter","","HLT")
)

patHLT2ElectronRelaxed = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter","","HLT")
)

# Muon triggers
patHLT1MuonIso = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltSingleMuIsoL3IsoFiltered","","HLT")
)

patHLT1MuonNonIso = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltSingleMuNoIsoL3PreFiltered","","HLT")
)

patHLT2MuonNonIso = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltDiMuonNoIsoL3PreFiltered","","HLT")
)

# BTau triggers
patHLT1Tau = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltFilterL3SingleTau","","HLT")
)

patHLT2TauPixel = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltFilterL25PixelTau","","HLT")
)

# JetMET triggers
patHLT2jet = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hlt2jet150","","HLT")
)

patHLT3jet = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hlt3jet85","","HLT")
)

patHLT4jet = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hlt4jet60","","HLT")
)

patHLT1MET65 = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hlt1MET65","","HLT")
)


## patTuple ##

# HLT_IsoMu11 - L1_SingleMu7
patHLTIsoMu11 = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltSingleMuIsoL3IsoFiltered","","HLT")
)

# HLT_Mu11 - L1_SingleMu7
patHLTMu11 = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltSingleMuNoIsoL3PreFiltered11","","HLT")
)

# HLT_DoubleIsoMu3 - L1_DoubleMu3
patHLTDoubleIsoMu3 = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("","","HLT")
)

# HLT_DoubleMu3 - L1_DoubleMu3
patHLTDoubleMu3 = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltDiMuonIsoL3IsoFiltered","","HLT")
)

# HLT_IsoEle15_LW_L1I - L1_SingleIsoEG12
patHLTIsoEle15LWL1I = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter","","HLT")
)

# HLT_Ele15_LW_L1R - L1_SingleEG10
patHLTEle15LWL1R = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter","","HLT")
)

# HLT_DoubleIsoEle10_LW_L1I - L1_DoubleIsoEG8
patHLTDoubleIsoEle10LWL1I = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter","","HLT")
)

# HLT_DoubleEle5_SW_L1R - L1_DoubleEG5
patHLTDoubleEle5SWL1R = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter","","HLT")
)

# HLT_LooseIsoTau_MET30_L1MET - L1_TauJet30_ETM30
patHLTLooseIsoTauMET30L1MET = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltFilterSingleTauMETEcalIsolationRelaxed","","HLT")
)

# HLT_DoubleIsoTau_Trk3 - L1_DoubleTauJet40
patHLTDoubleIsoTauTrk3 = cms.EDProducer("PATTrigProducer",
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    filterName = cms.InputTag("hltFilterL25PixelTau","","HLT")
)
           