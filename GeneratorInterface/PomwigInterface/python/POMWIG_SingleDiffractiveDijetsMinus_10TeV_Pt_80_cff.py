import FWCore.ParameterSet.Config as cms

herwig6Parameters = cms.PSet(
	comEnergy = cms.double(10000.0),
	useJimmy = cms.bool(False),
	doMPInteraction = cms.bool(False),

	herwigHepMCVerbosity = cms.untracked.bool(False),
	herwigVerbosity = cms.untracked.int32(1),
	printCards = cms.untracked.bool(True),
	maxEventsToPrint = cms.untracked.int32(2),

	crossSection = cms.untracked.double(-1.0),
	filterEfficiency = cms.untracked.double(1.0),
)

source = cms.Source("EmptySource")
 
generator = cms.EDFilter("PomwigGeneratorFilter",
    herwig6Parameters,
    VertexSmearing = cms.PSet(refToPSet_ = cms.string("VertexSmearingParameters")),
    HerwigParameters = cms.PSet(
        parameterSets = cms.vstring('SD2InclusiveJets'),
        SD2InclusiveJets = cms.vstring('NSTRU      = 14         ! H1 Pomeron Fit B', 
            'Q2WWMN     = 1E-6       ! Minimum |t|', 
            'Q2WWMX     = 4.0        ! Maximum |t|', 
            'YWWMIN     = 1E-6       ! Minimum xi', 
            'YWWMAX     = 0.2        ! Maximum xi', 
            'IPROC      = 11500      ! Process PomP -> jets', 
            'PTMIN      = 40         ! 2->2 PT min', 
            'MODPDF(1)  = 10150      ! Set MODPDF CTEQ61', 
            'MODPDF(2)  = -1         ! Set MODPDF')
    ),
    diffTopology = cms.int32(2),
    survivalProbability = cms.double(0.05),
    h1fit = cms.int32(2),
    doPDGConvert = cms.bool(False)
)

ProductionFilterSequence = cms.Sequence(generator)
