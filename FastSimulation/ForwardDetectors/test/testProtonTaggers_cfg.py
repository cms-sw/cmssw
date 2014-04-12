import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# ... the number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
# ... this is needed for the PtGun
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# ... this is needed for the PtGun
process.RandomNumberGeneratorService = cms.Service(
    "RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(123456781)
    ),
    sourceSeed = cms.untracked.uint32(123456781)
)

# ... this is needed in CMSSW >= 3_1
process.source = cms.Source("EmptySource")

# ... just a gun to feed something to the ProtonTaggerFilter
process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        # you can request more than 1 particle
        PartID = cms.vint32(2212),
        MinEta = cms.double(10.0),
        MaxEta = cms.double(10.4),
        MinPhi = cms.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.double(3.14159265359),
        MinPt = cms.double(0.4),
        MaxPt = cms.double(0.6)
    ),
    AddAntiParticle = cms.bool(False), ## back-to-back particles
    firstRun = cms.untracked.uint32(1),
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)
)

# ... put generator in ProductionFilterSequence (for CMSSW >= 3_1)
process.ProductionFilterSequence = cms.Sequence(process.generator)


# ... this is our forward proton filter
process.forwardProtonFilter = cms.EDFilter(
    "ProtonTaggerFilter",
    # ... choose where you want a proton to be detected for beam 1 (clockwise)
    #     0 -> ignore this beam
    #     1 -> only 420 (FP420)
    #     2 -> only 220 (TOTEM)
    #     3 -> 220 and 420 (region of overlay)
    #     4 -> 220 or 420 (combined acceptance)
    beam1mode = cms.uint32(4),

    # ... and for beam 2 (anti-clockwise)
    beam2mode = cms.uint32(1),

    # ... choose how the information for the two beam directions should be combined
    #     1 -> any of the two protons (clockwise or anti-clockwise) is enough
    #     2 -> both protons should be tagged
    #     3 -> two protons should be tagged as 220+220 or 420+420 (makes sence with beamXmode=4)
    #     4 -> two protons should be tagged as 220+420 or 420+220 (makes sence with beamXmode=4)
    beamCombiningMode = cms.uint32(1)
)

# ... request a summary to see how many events pass the filter
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# ... just run the filter
process.forwardProtons = cms.Path(process.ProductionFilterSequence * process.forwardProtonFilter)

# ...  define a root file for the events which pass the filter
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('forwardProtons'))
)

# ...  uncomment this if you want the output file
# process.saveIt = cms.EndPath(process.out)

