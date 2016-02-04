import FWCore.ParameterSet.Config as cms

HLTHiggsBits_ZZ = cms.EDAnalyzer("HLTHiggsBits",
  
    muon = cms.string('muons'),
    histName = cms.string(''),
    OutputMEsInRootFile = cms.bool(False),
 ##  histName = cms.string("#outputfile#"),
    Photon = cms.string('correctedPhotons'),
    genmet = cms.string('genMet'),
    MCTruth = cms.InputTag("genParticles"),
    hltBitNames = cms.vstring('HLT_Mu3','HLT_Mu9','HLT_DoubleMu3','HLT_Ele10_LW_L1R','HLT_Ele10_LW_EleId_L1R','HLT_Ele15_SW_LooseTrackIso_L1R','HLT_Ele15_SW_L1R','HLT_DoubleEle5_SW_L1R'),
    hltBitNamesEG = cms.vstring('HLT_Ele10_LW_L1R','HLT_Ele10_LW_EleId_L1R','HLT_Ele15_SW_L1R','HLT_Ele15_SW_LooseTrackIso_L1R','HLT_DoubleEle5_SW_L1R'),
    hltBitNamesMu = cms.vstring('HLT_Mu3','HLT_Mu9','HLT_DoubleMu3'),
    hltBitNamesPh = cms.vstring(''),
    hltBitNamesTau = cms.vstring(''),
   
   # Electron = cms.string('pixelMatchGsfElectrons'),
    Electron = cms.string('gsfElectrons'),
  
    Nchannel = cms.int32(1),   
    OutputFileName = cms.string(''),
    DQMFolder = cms.untracked.string("HLT/Higgs/HZZ"),
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT"),
    RunParameters = cms.PSet(
        Debug = cms.bool(True),
        Monte = cms.bool(True),
        EtaMax = cms.double(2.5),
        EtaMin = cms.double(-2.5)
    )
)


