import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")




process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:ttbar_5flavours_xqcut20_10TeV.lhe')
    # fileNames = cms.untracked.vstring('file:/uscms_data/d2/yarba_j/lhe_for_tests/7TeV_Zbb_run45040_unweighted_events_qcut13_mgPostv2.lhe')
    # fileNames = cms.untracked.vstring('file:/uscms_data/d2/yarba_j/lhe_for_tests/7TeV_ttbarjets_run621_unweighted_events_qcut40_mgPost.lhe')
)

process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
                        'PMAS(5,1)=4.4   ! b quark mass',
                        'PMAS(6,1)=172.4 ! t quark mass',
			'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
			'MSTP(61)=1      ! Parton showering on or off'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    ),
    jetMatching = cms.untracked.PSet(
       scheme = cms.string("Madgraph"),
       mode = cms.string("auto"),	# soup, or "inclusive" / "exclusive"
       MEMAIN_etaclmax = cms.double(5.), # -1. for other samples, to pick it up from LHE file
       MEMAIN_qcut = cms.double(30.),   # -1. for other samples, to pickup from LHE
       MEMAIN_minjets = cms.int32(-1),
       MEMAIN_maxjets = cms.int32(-1),
       MEMAIN_showerkt = cms.double(0),  # use 1=yes only for pt-ordered showers !
       MEMAIN_nqmatch = cms.int32(5), #PID of the flavor until which the QCD radiation are kept in the matching procedure; 
                                      # if nqmatch=4, then all showered partons from b's are NOT taken into account
				      # Note (JVY): for most cases it should stay 5 or can be set to -1 (if provided
				      #             in the LHE file; otherwise the job will throw); however, for bbar
				      #             it should be set to 4 (see above), unless it's given in the LHE
       MEMAIN_excres = cms.string(""),
       outTree_flag = cms.int32(0)        # 1=yes, write out the tree for future sanity check
    )    
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TestTTbar_MGmatch.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))

)

process.p = cms.Path(process.generator)
process.p1 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.p1, process.outpath)
