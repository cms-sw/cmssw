import FWCore.ParameterSet.Config as cms

HLTHiggsBits_gg = cms.EDAnalyzer("HLTHiggsBits",
  
    muon = cms.string('muons'),
    histName = cms.string(''),
    OutputMEsInRootFile = cms.bool(False),
 ##  histName = cms.string("#outputfile#"),
    Photon = cms.string('correctedPhotons'),
    MCTruth = cms.InputTag("genParticles"),
    hltBitNames = cms.vstring(''),
    hltBitNamesEG = cms.vstring(''),
    hltBitNamesMu = cms.vstring(''),
    hltBitNamesPh = cms.vstring('HLT_Photon15_L1R','HLT_Photon15_TrackIso_L1R','HLT_DoublePhoton10_L1R'),
    hltBitNamesTau = cms.vstring(''),
   
   # Electron = cms.string('pixelMatchGsfElectrons'),
    Electron = cms.string('gsfElectrons'), 
    Nchannel = cms.int32(3), 
  
    OutputFileName = cms.string(''),
    DQMFolder = cms.untracked.string("HLT/Higgs/Hgg"),
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT"),
    RunParameters = cms.PSet(
        Debug = cms.bool(True),
        Monte = cms.bool(True),
        EtaMax = cms.double(2.5),
        EtaMin = cms.double(-2.5)
    )
)


