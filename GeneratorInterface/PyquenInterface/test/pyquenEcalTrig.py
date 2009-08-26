import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")


from GeneratorInterface.PyquenInterface.pyquenPythiaDefault_cff import *
process.generator = cms.EDFilter("PyquenGeneratorFilter",
                         doQuench = cms.bool(True),
                         doIsospin = cms.bool(True),
                         qgpInitialTemperature = cms.double(1.0), ## initial temperature of QGP; allowed range [0.2,2.0]GeV;
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         doRadiativeEnLoss = cms.bool(True), ## if true, perform partonic radiative en loss
                         bFixed = cms.double(0.0), ## fixed impact param (fm); valid only if cflag_=0
                         angularSpectrumSelector = cms.int32(0), ## angular emitted gluon spectrum :
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     parameterSets = cms.vstring('pythiaDefault','pythiaJets','pythiaPromptPhotons','kinematics'),
                                                     kinematics = cms.vstring('CKIN(3) = 50','CKIN(4) = 80')
                                                     ),
                         qgpProperTimeFormation = cms.double(0.1), ## proper time of QGP formation; allowed range [0.01,10.0]fm/c;
                         # Center of mass energy
                         comEnergy = cms.double(4000.0),
                         
                         qgpNumQuarkFlavor = cms.int32(0), ## number of active quark flavors in qgp; allowed values: 0,1,2,3
                         cFlag = cms.int32(0), ## centrality flag
                         bMin = cms.double(0.0), ## min impact param (fm); valid only if cflag_!=0
                         bMax = cms.double(0.0), ## max impact param (fm); valid only if cflag_!=0
                         maxEventsToPrint = cms.untracked.int32(0), ## events to print if pythiaPylistVerbosit
                         aBeamTarget = cms.double(208.0), ## beam/target atomic number
                         doCollisionalEnLoss = cms.bool(True), ## if true, perform partonic collisional en loss

                         embeddingMode = cms.bool(True),
                         filterType = cms.untracked.string("EcalCandidateSkimmer"),

                         partons = cms.vint32(1,2,3,4,5,6),
                         partonStatus = cms.vint32(3,3,3,3,3,3),
                         partonPt = cms.vdouble(50,50,50,50,50,50),

                         particles = cms.vint32(211,321,221),
                         particleStatus = cms.vint32(1,1,1),
                         particlePt = cms.vdouble(50,50,50),
                         etaMax = cms.double(2)                         

                         )







process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10)
                                       )

process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )

process.ana = cms.EDAnalyzer('HydjetAnalyzer'
                             )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('treefile.root')
                                   )

process.p = cms.Path(process.generator*process.ana)




