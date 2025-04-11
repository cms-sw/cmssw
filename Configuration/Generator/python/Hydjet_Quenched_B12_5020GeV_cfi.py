import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pyquen2025Settings_cff import *
from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
import os

hjenergy = os.getenv("HJENERGY", "5020")

generator = ExternalGeneratorFilter(cms.EDFilter("HydjetGeneratorFilter",
                         locals()[f"collisionParameters{hjenergy}GeV"],   #tune CELLO
                         locals()[f"qgpParameters{hjenergy}GeV"],         #tune CELLO
                         locals()[f"hydjetParameters{hjenergy}GeV"],      #tune CELLO
                         hydjetMode = cms.string('kHydroQJets'),
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     # Quarkonia and Weak Bosons added back upon dilepton group's request.
                                                     parameterSets = cms.vstring('pythiaUESettings',
                                                                                 'hydjetPythiaDefault'+hjenergy, #tune CELLO
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
                         bMax = cms.double(22),
                         bFixed = cms.double(12)
                         ))
