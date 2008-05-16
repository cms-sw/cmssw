import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTGRAPHDUMPER")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.ecalPedHists = cms.EDFilter("EcalPedHists",
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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    #untracked uint32 skipEvents = 16000
    #untracked vstring fileNames = {'file:/data/scooper/data/P5_Co-07/P5_Co.00027909.A.0.0.root'}
    fileNames = cms.untracked.vstring('file:/data/scooper/data/grea07/40792E58-B757-DC11-8AB2-001617E30F46.root')
)

process.p = cms.Path(process.ecalEBunpacker*process.ecalPedHists)

