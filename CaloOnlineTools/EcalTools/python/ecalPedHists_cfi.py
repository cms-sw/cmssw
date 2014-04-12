import FWCore.ParameterSet.Config as cms

ecalPedHists = cms.EDAnalyzer("EcalPedHists",
    # sepecify list of samples to use
    listSamples = cms.untracked.vint32(1, 2, 3),
    EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    # selection on EB+- numbering
    listEBs = cms.untracked.vstring(),
    # specify list of channels to be dumped
    # outputs all channels if not specified
    listChannels = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10),
    fileName = cms.untracked.string('ecalPedHists'),
    # selection on FED number (601...654); -1 selects all 
    listFEDs = cms.untracked.vint32(-1),
    headerProducer = cms.InputTag("ecalEBunpacker")
)


