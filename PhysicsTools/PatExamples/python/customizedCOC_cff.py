import FWCore.ParameterSet.Config as cms


cocPatJets = cms.EDProducer("PATJetCleaner",
    src = cms.InputTag("selectedPatJets"), 

    # preselection (any string-based cut on pat::Jet)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(

        isolatedMuons = cms.PSet(
           src       = cms.InputTag("isolatedPatMuons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the jet to be discared
        ),                                
        isolatedElectrons = cms.PSet(
           src       = cms.InputTag("isolatedPatElectrons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the jet to be discared
        )
    ),
    # finalCut (any string-based cut on pat::Jet)
    finalCut = cms.string(''),
)

customCOC = cms.Sequence(
     cocPatJets
)
