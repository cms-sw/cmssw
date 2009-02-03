import FWCore.ParameterSet.Config as cms

from GeneratorInterface.PyquenInterface.pyquenPythiaDefault_cff import *
source = cms.Source("HydjetSource",

                        maxEventsToPrint = cms.untracked.int32(0),
                        pythiaPylistVerbosity = cms.untracked.int32(0),

                        firstEvent = cms.untracked.uint32(1),
                        firstRun = cms.untracked.uint32(1),
                    
                        comEnergy = cms.double(5500.0),
                        sigmaInelNN = cms.double(58),
                        aBeamTarget = cms.double(208.0),

                        nMultiplicity = cms.int32(26000),

                        hydjetMode = cms.string('kHydroQJets'),

                        # Valid entries:
                        # kHydroOnly  //jet production off (pure HYDRO event): nhsel=0
                        # kHydroJets  //jet production on, jet quenching off (HYDRO+njet*PYTHIA events):nhsel=1
                        # kHydroQJets //jet production & jet quenching on (HYDRO+njet*PYQUEN events);nhsel=2
                        # kJetsOnly   //jet production on, jet quenching off, HYDRO off (njet*PYTHIA events):nhsel=3
                        # kQJetsOnly  //jet production & jet quenching on, HYDRO off (njet*PYQUEN events):nhsel=4

                        shadowingSwitch = cms.int32(0),
                        doRadiativeEnLoss = cms.bool(True),
                        doCollisionalEnLoss = cms.bool(True),
                        rotateEventPlane = cms.bool(True),
                        fracSoftMultiplicity = cms.double(1.),
                        hadronFreezoutTemperature = cms.double(0.14),
                        maxLongitudinalRapidity = cms.double(3.75),
                        maxTransverseRapidity = cms.double(1.),
                        qgpNumQuarkFlavor = cms.int32(0),
                        qgpInitialTemperature = cms.double(1.),
                        qgpProperTimeFormation = cms.double(0.1),
                    PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                parameterSets = cms.vstring('pythiaDefault','csa08Settings','pythiaJets','pythiaPromptPhotons')
                                                ),
                    cFlag = cms.int32(0),
                    bFixed = cms.double(0),
                    bMin = cms.double(0),
                    bMax = cms.double(0),
                    
                    allowEmptyEvents = cms.bool(False)
                    )

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.6 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/GeneratorInterface/HydjetInterface/python/hydjetSourceDefault_cfi.py,v $'),
    annotation = cms.untracked.string('PYTHIA6-MinBias at 10TeV')
    )





