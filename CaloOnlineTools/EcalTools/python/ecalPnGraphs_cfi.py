import FWCore.ParameterSet.Config as cms

ecalPnGraphs = cms.EDAnalyzer("EcalPnGraphs",
    # requested EBs 
    requestedEbs = cms.untracked.vstring('none'),
    # length of the line centered on listPns containing the Pns you want to see
    # needs to be an odd number
    numPn = cms.untracked.int32(9),
    fileName = cms.untracked.string('test.root'),
    digiProducer = cms.string('ecalEBunpacker'),
    # requested FEDs
    requestedFeds = cms.untracked.vint32(-1),
    # list of Pns to be graphed around
    listPns = cms.untracked.vint32(5)
)


