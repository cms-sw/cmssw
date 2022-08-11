import FWCore.ParameterSet.Config as cms
process = cms.Process("Analysis")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
#process.load("GeneratorInterface.Hydjet2Interface.hydjet2Default_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/user/mnguyen//hydjet/MinBias_Hydjet_Drum5F_5p02TeV/hydjet_fromCentralAOD_noGenCut_forest/merged_HiForestMiniAOD.root'

#	'/store/himc/Run3Winter22PbPbNoMixGS/MinBias_Hydjet_Drum5F_5p02TeV/GEN-SIM/122X_mcRun3_2021_realistic_HI_v10-v1/2530000/000343c8-f986-4144-8573-7ba8b6df0b8b.root'
    )
)


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.ana = cms.EDAnalyzer('Hydjet2Analyzer',

                src = cms.untracked.InputTag("generatorSmeared"),
		doHistos = cms.untracked.bool(True),
                userHistos = cms.untracked.bool(False),
		doAnalysis = cms.untracked.bool(True),
                doTestEvent = cms.untracked.bool(False), # for debuging event output information

		###Settings for USER histos

		#status
                uStatus = cms.untracked.int32(1), #1 - it's 1,2,3,4,5 of Pythia status; 2 - 11,12,13,14,15; 3 - All

		#up to 3 abs(PDG) for selection, if less needed just comment not used
                uPDG_1 = cms.untracked.int32(211),
                uPDG_2 = cms.untracked.int32(321),
                uPDG_3 = cms.untracked.int32(2212),

                # |eta| cut for pT dep.dist.
                dPTetaCut = cms.untracked.double(0.), #down
                uPTetaCut = cms.untracked.double(0.8), #up

                #Vectors of bins borders(when 0 - uniform bins would be used)
                PtBins = cms.untracked.vdouble(0.), #, 1., 2., 3., 4., 5., 6., 8., 12., 16., 20.),
                EtaBins = cms.untracked.vdouble(0.),
                PhiBins = cms.untracked.vdouble(0.),
                v2EtaBins = cms.untracked.vdouble(0.),
                v2PtBins = cms.untracked.vdouble(0.), #, 1., 2., 3., 4., 6., 8., 12., 16., 20.),

                #Settings for uniform bins 
		nintPt 		= cms.untracked.int32(100),
                nintEta 	= cms.untracked.int32(51),
		nintPhi		= cms.untracked.int32(100),
		nintV2pt	= cms.untracked.int32(100),
		nintV2eta	= cms.untracked.int32(100),

		minPt		= cms.untracked.double(0.),
		minEta		= cms.untracked.double(-10.),
		minPhi		= cms.untracked.double(-3.14159265358979),
		minV2pt		= cms.untracked.double(0.),
		minV2eta	= cms.untracked.double(-10.),
	
		maxPt		= cms.untracked.double(100.),
		maxEta		= cms.untracked.double(10.),
		maxPhi		= cms.untracked.double(3.14159265358979),
		maxV2pt		= cms.untracked.double(10.),
		maxV2eta	= cms.untracked.double(10.),

)

#to separate hydro and jet parts of hydjet2	
#process.generator.separateHydjetComponents = cms.untracked.bool(False)
Debug = None

if Debug:
	process.load("FWCore.MessageLogger.MessageLogger_cfi")

	process.MessageLogger = cms.Service("MessageLogger",

		destinations     = cms.untracked.vstring('LogDebug_Hydjet2'),
		categories       = cms.untracked.vstring(
                	                        	'Hydjet2',
                        	                	'Hydjet2_array'
                                		        ),
		LogDebug_Hydjet2 = cms.untracked.PSet(
        		threshold =  cms.untracked.string('DEBUG'), #Priority: DEBUG < INFO < WARNING < ERROR
        		DEBUG   = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
        		INFO    = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        		WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        		ERROR   = cms.untracked.PSet(limit = cms.untracked.int32(0)),
			#Categores
        		Hydjet2  = cms.untracked.PSet(
                        	limit = cms.untracked.int32(-1), # number of masseges 
                        	timespan = cms.untracked.int32(0)     #time to resete limit counter in seconds
                        	),

        		Hydjet2_array  = cms.untracked.PSet(
                        	limit = cms.untracked.int32(-1), # number of masseges 
                        	timespan = cms.untracked.int32(0)     #time to resete limit counter in seconds
                        	)

			),
			debugModules     = cms.untracked.vstring('*')
		)

process.TFileService = cms.Service('TFileService',
	fileName = cms.string('Hydjet1GS.root')
)

process.p = cms.Path(process.ana)
