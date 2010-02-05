import FWCore.ParameterSet.Config as cms

cleanLayer1Muons = cms.EDFilter("PATMuonCleaner",
    src = cms.InputTag("selectedLayer1Muons"), 

    # preselection (any string-based cut for pat::Muon)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(),

    # finalCut (any string-based cut for pat::Muon)
    finalCut = cms.string(''),
)
