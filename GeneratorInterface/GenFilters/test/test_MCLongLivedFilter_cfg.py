import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")
process.load("FWCore.MessageService.MessageLogger_cfi")

# control point for all seeds
#
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("GeneratorInterface.Pythia6Interface.pythiaDefault_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string(''),
    name = cms.untracked.string(''),
    annotation = cms.untracked.string('generation of D*, with LongLived filter applied')
)
process.source = cms.Source("EmptySource")
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    Ptmax = cms.untracked.double(200.0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    ymax = cms.untracked.double(10.0),
    ParticleID = cms.untracked.int32(413),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    DoubleParticle = cms.untracked.bool(False),
    Ptmin = cms.untracked.double(200.0),
    ymin = cms.untracked.double(-10.0),
    maxEventsToPrint = cms.untracked.int32(0),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        process.pythiaDefaultBlock,
        # User cards - name is "myParameters"                                
        # Pythia's random generator initialization                           
        myParameters = cms.vstring('MDCY(123,2) = 738', 
            'MDCY(123,3) = 1', 
            'MDCY(122,2) = 705', 
            'MDCY(122,3) = 1'),
        # This is a vector of ParameterSet names to be read, in this order   
        # The first two are in the include files below                       
        # The last one are simply my additional parameters                   
        parameterSets = cms.vstring('pythiaDefault', 
            'myParameters')
    )
)

process.select = cms.EDFilter("MCLongLivedParticles",
    hepMCProductTag = cms.InputTag("generator"),
    LengCut = cms.untracked.double(100.0) ## in mm

)

process.out = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('dstardecay.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN')
    )
)

process.p1 = cms.Path(process.generator*process.select)
process.outpath = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.p1,process.outpath)

process.generator.pythiaPylistVerbosity = 0
process.generator.maxEventsToPrint = 10
process.generator.pythiaHepMCVerbosity = True


