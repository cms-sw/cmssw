import FWCore.ParameterSet.Config as cms

from GeneratorInterface.PyquenInterface.pyquenPythiaDefault_cff import *
hydjetDefaultParameters = cms.PSet(
                    
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
                                                # Quarkonia and Weak Bosons removed upon dilepton group's request.
                                                parameterSets = cms.vstring('pythiaDefault','pythiaJets','pythiaPromptPhotons'),
                                                ),
                    
                    allowEmptyEvents = cms.bool(False)
                    )




