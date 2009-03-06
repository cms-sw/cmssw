# The following comments couldn't be translated into the new config version:

# MSEL=1 (hard QCD dijets in) + hydjet needed initializations
import FWCore.ParameterSet.Config as cms

from GeneratorInterface.PyquenInterface.pyquenPythiaDefault_cfi import *
source = cms.Source("HydjetSource",
    shadowingSwitch = cms.int32(1), ## nuclear shadowing 1=ON, 0=OFF

    # range: [0.01,7.0]
    maxTransverseRapidity = cms.double(1.5),
    # =  0 fixed impact param
    # <> 0 between bmin and bmax
    comEnergy = cms.double(5500.0),
    # allowed range [0.01,10.0]fm/c;
    sigmaInelNN = cms.double(58.0),
    doRadiativeEnLoss = cms.bool(True), ## if true, perform partonic radiative en loss

    # allowed values: 0,1,2,3	
    qgpInitialTemperature = cms.double(1.0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    aBeamTarget = cms.double(207.0), ## beam/target atomic number

    # range: 0<bmin<bmax
    cFlag = cms.int32(0),
    # range: [0.08,0.2]GeV
    hydjetMode = cms.string('kHydroQJets'),
    hadronFreezoutTemperature = cms.double(0.14), ## hadron freez-out temperature

    # range: [0.01,3.0]
    nMultiplicity = cms.int32(26000),
    # automatically calculated for other centralities and beams
    qgpNumQuarkFlavor = cms.int32(0),
    doCollisionalEnLoss = cms.bool(True), ## if true, perform partonic collisional en loss

    bFixed = cms.double(0.0), ## fixed impact param (fm); 

    # Valid entries:
    # kHydroOnly  //jet production off (pure HYDRO event): nhsel=0
    # kHydroJets  //jet production on, jet quenching off (HYDRO+njet*PYTHIA events):nhsel=1
    # kHydroQJets //jet production & jet quenching on (HYDRO+njet*PYQUEN events);nhsel=2
    # kJetsOnly   //jet production on, jet quenching off, HYDRO off (njet*PYTHIA events):nhsel=3
    # kQJetsOnly  //jet production & jet quenching on, HYDRO off (njet*PYQUEN events):nhsel=4
    maxLongitudinalRapidity = cms.double(4.0),
    # range: bmin<bmax<3*RA
    bMin = cms.double(0.0),
    fracSoftMultiplicity = cms.double(1.0), ## fraction of soft hydro induced hadronic multiplicity

    maxEventsToPrint = cms.untracked.int32(0), ## events to print if pythiaPylistVerbosit

    # valid only if cflag_=0
    # range: 0<bFixed<3*RA
    bMax = cms.double(0.0),
    PythiaParameters = cms.PSet(
        pyquenPythiaDefaultBlock,
        # This is a vector of ParameterSet names to be read, in this order
        # They are  in the include files
        # pythiaDefault HAS TO BE ALWAYS included
        # If just the hard QCD dijets wanted, comment out the photons corresponding include and parameter set
        parameterSets = cms.vstring('pythiaDefault')
    ),
    # allowed range [0.2,2.0]GeV;
    qgpProperTimeFormation = cms.double(0.1)
)


