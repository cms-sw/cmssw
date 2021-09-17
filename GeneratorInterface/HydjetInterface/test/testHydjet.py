import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("GeneratorInterface.HydjetInterface.hydjetDefault_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
	generator = cms.PSet(
		initialSeed = cms.untracked.uint32(123456789),
		engineName = cms.untracked.string('HepJamesRandom')
	)
)

process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(5)
	)

process.ana = cms.EDAnalyzer('HydjetAnalyzer',

	doHistos		= cms.untracked.bool(True),
	userHistos		= cms.untracked.bool(False),
        
	# Settings for USER histos

	uStatus		= cms.untracked.int32(2),	# 1 - it's 1,2,3,4,5 of Pythia status; 2 - 11,12,13,14,15; 3 - All
	uPDG_1		= cms.untracked.int32(443),
	uPDG_2		= cms.untracked.int32(-443),

	### Eta cut for pT dep. dist.
	uPTetaCut	= cms.untracked.double(4.),
	dPTetaCut	= cms.untracked.double(2.5),

	### Vectors of bins borders (when 0 -  uniform bins would be used)
	PtBins 		= cms.untracked.vdouble(0.,1.,2.,3.,4.,5.,6.,8.,12.,16.,20.),
	EtaBins 		= cms.untracked.vdouble(0.),
	PhiBins 		= cms.untracked.vdouble(0.),
	v2EtaBins		= cms.untracked.vdouble(0.),
	v2PtBins 		= cms.untracked.vdouble(0.,1.,2.,3.,4.,6.,8.,12.,16.,20.),
	
	### Settings for uniform bins
	nintPt		= cms.untracked.int32(1000),
	nintEta		= cms.untracked.int32(100),
	nintPhi		= cms.untracked.int32(100),
	nintV2pt		= cms.untracked.int32(100),
	nintV2eta		= cms.untracked.int32(100),

	minPt		= cms.untracked.double(0.),
	minEta		= cms.untracked.double(-10.),
	minPhi		= cms.untracked.double(-3.14159265358979),
	minV2pt		= cms.untracked.double(0.),
	minV2eta		= cms.untracked.double(-10.),
	
	maxPt		= cms.untracked.double(100.),
	maxEta		= cms.untracked.double(10.),
	maxPhi		= cms.untracked.double(3.14159265358979),
	maxV2pt		= cms.untracked.double(10.),
	maxV2eta		= cms.untracked.double(10.),

)


#process.generator.signalVtx = cms.untracked.vdouble(0.,0.,0.,0.) # Signal event vertex option, to set it by hand (instead of smearing)

process.TFileService = cms.Service('TFileService',
	fileName = cms.string('Hydjet1_MB_5020GeV.root')
)

process.p = cms.Path(process.generator*process.ana)


