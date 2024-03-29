import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pyquen2015Settings_cff import *

_generator = cms.EDFilter("HydjetGeneratorFilter",
                         collisionParameters5020GeV,
                         qgpParameters,
                         hydjetParameters,
                         hydjetMode = cms.string('kHydroQJets'),
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     # Quarkonia and Weak Bosons added back upon dilepton group's request.
                                                     parameterSets = cms.vstring('pythiaUESettings',
                                                                                 'hydjetPythiaDefault',
                                                                                 'myParameters',
                                                                                 'pythiaJets',
                                                                                 'pythiaPromptPhotons',
                                                                                 'pythiaZjets',
                                                                                 'pythiaBottomoniumNRQCD',
                                                                                 'pythiaCharmoniumNRQCD',
                                                                                 'pythiaQuarkoniaSettings',
                                                                                 'pythiaWeakBosons'
                                                                                 )
                                                     ),
                         cFlag = cms.int32(0),
                         bMin = cms.double(0),
                         bMax = cms.double(30),
                         bFixed = cms.double(12)
                         )

from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
generator = ExternalGeneratorFilter(_generator)
