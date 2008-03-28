import FWCore.ParameterSet.Config as cms

hltLevel1GTSeed = cms.EDFilter("HLTLevel1GTSeed",
    # logical expression for the required L1 algorithms;
    # the algorithms are specified by name
    # allowed operators: "AND", "OR", "NOT", "(", ")"
    #
    # by convention, "L1GlobalDecision" logical expression means global decision
    # 
    L1SeedsLogicalExpression = cms.string(''),
    # InputTag for L1 muon collection
    L1MuonCollectionTag = cms.InputTag("l1extraParticles"),
    # InputTag for the L1 Global Trigger DAQ readout record
    #   GT Emulator = gtDigis
    #   GT Unpacker = l1GtUnpack
    #
    #   cloned GT unpacker in HLT = gtDigis
    L1GtReadoutRecordTag = cms.InputTag("gtDigis"),
    # InputTag for L1 particle collections (except muon)
    #   L1 Extra = l1extraParticles
    #
    L1CollectionsTag = cms.InputTag("l1extraParticles"),
    # InputTag for L1 Global Trigger object maps
    #   only the emulator produces the object maps
    #   GT Emulator = gtDigis
    #
    #   cloned GT emulator in HLT = l1GtObjectMap
    #
    L1GtObjectMapTag = cms.InputTag("l1GtObjectMap"),
    # seeding done via technical trigger bits, if value is "true";
    # default: false (seeding via physics algorithms)
    L1TechTriggerSeeding = cms.bool(False)
)


