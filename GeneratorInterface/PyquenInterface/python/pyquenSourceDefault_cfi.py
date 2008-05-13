# The following comments couldn't be translated into the new config version:

# MSEL=1 (hard QCD in) + pyquen needed initializations
import FWCore.ParameterSet.Config as cms

from GeneratorInterface.PyquenInterface.pyquenPythiaDefault_cfi import *
source = cms.Source("PyquenSource",
    # 0-small angle, 1-broad angle, 2-collinear                           
    doQuench = cms.bool(True),
    qgpInitialTemperature = cms.double(1.0), ## initial temperature of QGP; allowed range [0.2,2.0]GeV;

    pythiaPylistVerbosity = cms.untracked.int32(0),
    doRadiativeEnLoss = cms.bool(True), ## if true, perform partonic radiative en loss

    bFixed = cms.double(0.0), ## fixed impact param (fm); valid only if cflag_=0

    angularSpectrumSelector = cms.int32(0), ## angular emitted gluon spectrum : 

    pythiaHepMCVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        pyquenPythiaDefaultBlock,
        # This is a vector of ParameterSet names to be read, in this order
        # They are  in the include files 
        # pythiaDefault HAS TO BE ALWAYS included
        # If just the hard QCD dijets wanted, comment out the photons corresponding include and parameter set
        parameterSets = cms.vstring('pythiaDefault')
    ),
    qgpProperTimeFormation = cms.double(0.1), ## proper time of QGP formation; allowed range [0.01,10.0]fm/c; 

    # =  0 fixed impact param
    # <> 0 --> minbias with standard glauber geometry
    comEnergy = cms.double(5500.0),
    numQuarkFlavor = cms.int32(0), ## number of active quark flavors in qgp; allowed values: 0,1,2,3

    cFlag = cms.int32(0), ## centrality flag

    maxEventsToPrint = cms.untracked.int32(0), ## events to print if pythiaPylistVerbosit

    aBeamTarget = cms.double(207.0), ## beam/target atomic number

    doCollisionalEnLoss = cms.bool(True) ## if true, perform partonic collisional en loss

)


