# The following comments couldn't be translated into the new config version:

# MSEL=1 (hard QCD in) + pyquen needed initializations
import FWCore.ParameterSet.Config as cms

from GeneratorInterface.PyquenInterface.pyquenPythiaDefault_cff import *
generator = cms.EDFilter("PyquenGeneratorFilter",
                         doQuench = cms.bool(True),
                         doIsospin = cms.bool(True),
                         qgpInitialTemperature = cms.double(1.0), ## initial temperature of QGP; allowed range [0.2,2.0]GeV;
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         doRadiativeEnLoss = cms.bool(True), ## if true, perform partonic radiative en loss
                         bFixed = cms.double(0.0), ## fixed impact param (fm); valid only if cflag_=0
                         angularSpectrumSelector = cms.int32(0), ## angular emitted gluon spectrum :
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     parameterSets = cms.vstring('pyquenMain','pythiaUESettings','ppJets','pythiaPromptPhotons','kinematics'),
                                                     kinematics = cms.vstring('CKIN(3) = 50','CKIN(4) = 80')
                                                     ),
                         qgpProperTimeFormation = cms.double(0.1), ## proper time of QGP formation; allowed range [0.01,10.0]fm/c;
                         # Center of mass energy
                         comEnergy = cms.double(2800.0),

                         qgpNumQuarkFlavor = cms.int32(0), ## number of active quark flavors in qgp; allowed values: 0,1,2,3
                         cFlag = cms.int32(0), ## centrality flag
                         bMin = cms.double(0.0), ## min impact param (fm); valid only if cflag_!=0
                         bMax = cms.double(0.0), ## max impact param (fm); valid only if cflag_!=0
                         maxEventsToPrint = cms.untracked.int32(0), ## events to print if pythiaPylistVerbosit
                         aBeamTarget = cms.double(208.0), ## beam/target atomic number
                         doCollisionalEnLoss = cms.bool(False), ## if true, perform partonic collisional en loss
                         embeddingMode = cms.bool(False)
                         
                         )


