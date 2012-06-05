import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:../../Pythia6Interface/test/ttbar_5flavours_xqcut20_10TeV.lhe')

)

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.),
    jetMatching = cms.untracked.PSet(
       scheme = cms.string("Madgraph"),
       mode = cms.string("auto"),	# soup, or "inclusive" / "exclusive"
       MEMAIN_etaclmax = cms.double(5.0),
       MEMAIN_qcut = cms.double(30.0),
       MEMAIN_minjets = cms.int32(-1),
       MEMAIN_maxjets = cms.int32(-1),
       MEMAIN_showerkt = cms.double(0), # use 1=yes only for pt-ordered showers !
       MEMAIN_nqmatch = cms.int32(5),   #PID of the flavor until which the QCD radiation are kept in the matching procedure; 
                                        # if nqmatch=4, then all showered partons from b's are NOT taken into account
				        # Note (JY): I think the default should be 5 (b); anyway, don't try -1  as it'll result in a throw...
       MEMAIN_excres = cms.string(""),
       outTree_flag = cms.int32(0)      # 1=yes, write out the tree for future sanity check
    ),    
    PythiaParameters = cms.PSet(
        pythia8_mg = cms.vstring(''),
        parameterSets = cms.vstring('pythia8_mg')
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(123456),
        g4SimHits = cms.untracked.uint32(123456788),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py8Had_mgmatching.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

