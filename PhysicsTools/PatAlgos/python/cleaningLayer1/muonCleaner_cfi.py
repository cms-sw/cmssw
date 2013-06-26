import FWCore.ParameterSet.Config as cms

cleanPatMuons = cms.EDProducer("PATMuonCleaner",
    src = cms.InputTag("selectedPatMuons"), 

    # preselection (any string-based cut for pat::Muon)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(),

    # finalCut (any string-based cut for pat::Muon)
    finalCut = cms.string(''),
)
