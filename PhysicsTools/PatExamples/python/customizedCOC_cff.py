import FWCore.ParameterSet.Config as cms

cocPatElectrons = cms.EDProducer("PATElectronCleaner",
    ## pat electron input source
    src = cms.InputTag("isolatedPatElectrons"), 

    # preselection (any string-based cut for pat::Electron)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
           src       = cms.InputTag("selectedPatMuons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),  # don't preselect the muons
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
        isolatedMuons = cms.PSet(
           src       = cms.InputTag("isolatedPatMuons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),  # don't preselect the muons
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the electron to be discared
        )                                     
    ),

    # finalCut (any string-based cut for pat::Electron)
    finalCut = cms.string(''),
)

cocPatJets = cms.EDProducer("PATJetCleaner",
    src = cms.InputTag("selectedPatJets"), 

    # preselection (any string-based cut on pat::Jet)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
           src       = cms.InputTag("selectedPatMuons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the jet to be discared
        ),
        isolatedMuons = cms.PSet(
           src       = cms.InputTag("isolatedPatMuons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the jet to be discared
        ),                                
        electrons = cms.PSet(
           src       = cms.InputTag("selectedPatElectrons"),
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
        ),                                
        cocElectrons = cms.PSet(
           src       = cms.InputTag("cocPatElectrons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the jet to be discared
        ),
    ),
    # finalCut (any string-based cut on pat::Jet)
    finalCut = cms.string(''),
)

customCOC = cms.Sequence(
    cocPatElectrons * cocPatJets
)
