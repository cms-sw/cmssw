import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:../../Pythia6Interface/test/ttbar_5flavours_xqcut20_10TeV.lhe')
    # fileNames = cms.untracked.vstring('file:/storage/local/data1/condor/mrenna/lhe/7TeV_Zbb_run45040_unweighted_events_qcut13_mgPostv2.lhe')
    # fileNames = cms.untracked.vstring('file:/uscmst1b_scratch/lpc1/3DayLifetime/recovery_for_julia/storage/local/data1/condor/mrenna/lhe/7TeV_ttbarjets_run621_unweighted_events_qcut40_mgPost.lhe')
    # fileNames = cms.untracked.vstring('file:/storage/local/data1/condor/mrenna/lhe/7TeV_avjets_run50000_unweighted_events_qcut15_mgPost.lhe')
    # fileNames = cms.untracked.vstring('file:/storage/local/data1/condor/mrenna/lhe/7TeV_zvv_200_HT_inf_run114000_unweighted_events_qcut20_mgPostv2.lhe')
    # fileNames = cms.untracked.vstring('/store/user/mrenna/7TeV_ZbbToLL_M_50_run1001to1018_3processes_unweighted_events.lhe')
)


# process.load("Configuration.Generator.Hadronizer_MgmMatchTune4C_7TeV_madgraph_pythia8_cff")

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.),
    jetMatching = cms.untracked.PSet(
       scheme = cms.string("Madgraph"),
       mode = cms.string("auto"),	# soup, or "inclusive"/"exclusive"
       #
       # ATTENTION PLEASE !
       # One can set some parameters to -1 to make the tool pock it up from LHE file.
       # However, -1 is ONLY possible if a givcen parameter is present in LHE file
       # - otherwise the code will throw. 
       # So the user should make sure what it is and what she/he wants to do.
       #
       MEMAIN_etaclmax = cms.double(5.),
       MEMAIN_qcut = cms.double(30.),       
       MEMAIN_minjets = cms.int32(-1),
       MEMAIN_maxjets = cms.int32(-1),
       MEMAIN_showerkt = cms.double(0),    # use 1=yes only for pt-ordered showers !
       MEMAIN_nqmatch = cms.int32(5),      # PID of the flavor until which the QCD radiation are kept in the matching procedure. 
                                           # If nqmatch=4, then all showered partons from b's are NOT taken into account.
				           # In many cases the D=5
       MEMAIN_excres = cms.string(""),
       outTree_flag = cms.int32(0)         # 1=yes, write out the tree for future sanity check
    ),    
    PythiaParameters = cms.PSet(
        pythia8_mg = cms.vstring(''), # this pset is for very initial testing
        # this pset below is actually used in large-scale (production-type) tests
	processParameters = cms.vstring(
            'Main:timesAllowErrors    = 10000', 
        'ParticleDecays:limitTau0 = on',
            'ParticleDecays:tauMax = 10',
	    # '15:onMode = off', # tmp turn off tau decays, to process av sample (crash in Tau::decay in Py8)
        'Tune:ee 3',
        'Tune:pp 5',
	'ParticleDecays:sophisticatedTau = 0' ),
        parameterSets = cms.vstring('processParameters')
    )
)

#process.MessageLogger = cms.Service("MessageLogger",
#    cout = cms.untracked.PSet(
#        default = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        )
#    ),
#    destinations = cms.untracked.vstring('cout')
#)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(123456),
        g4SimHits = cms.untracked.uint32(123456788),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py8Had_mgmatching.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

#process.schedule = cms.Schedule(process.p, process.outpath)
process.schedule = cms.Schedule(process.p)

