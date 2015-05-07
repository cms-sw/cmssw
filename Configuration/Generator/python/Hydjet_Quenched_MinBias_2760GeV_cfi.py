import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pyquen2013Settings_cff import *

generator = cms.EDFilter("HydjetGeneratorFilter",
                         collisionParameters,
                         qgpParameters,
                         hydjetParameters,
                         hydjetMode = cms.string('kHydroQJets'),
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     # Quarkonia and Weak Bosons added back upon dilepton group's request.
                                                     parameterSets = cms.vstring('pythiaUESettings',
                                                                                 'hydjetPythiaDefault',
                                                                                 'pythiaJets',
                                                                                 'pythiaPromptPhotons',
                                                                                 'pythiaZjets',
                                                                                 'pythiaBottomoniumNRQCD',
                                                                                 'pythiaCharmoniumNRQCD',
                                                                                 'pythiaQuarkoniaSettings',
                                                                                 'pythiaWeakBosons'
                                                                                 )
                                                     ),
                         cFlag = cms.int32(1),
                         bMin = cms.double(0),
                         bMax = cms.double(30),
                         bFixed = cms.double(0)
                         )

