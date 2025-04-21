import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pyquen2025Settings_cff import *
from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
import os

hjenergy = os.getenv("HJENERGY", "2760")

generator = ExternalGeneratorFilter(cms.EDFilter("PyquenGeneratorFilter",
                         locals()[f"collisionParameters{hjenergy}GeV"],   #tune CELLO
                         locals()[f"qgpParameters{hjenergy}GeV"],         #tune CELLO
                         pyquenParameters,
                         doQuench = cms.bool(True),
                         bFixed = cms.double(0.0), ## fixed impact param (fm); valid only if cflag_=0
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     parameterSets = cms.vstring('pythiaUESettings','customProcesses','pythiaPromptPhotons','kinematics'),
                                                     kinematics = cms.vstring ("CKIN(3)=20",  #min pthat
                                                                               "CKIN(4)=9999", #max pthat
                                                                               "CKIN(7)=-2.",  #min rapidity
                                                                               "CKIN(8)=2."    #max rapidity
                                                                               )
                                                     
                                                     ),
                         cFlag = cms.int32(0), ## centrality flag
                         bMin = cms.double(0.0), ## min impact param (fm); valid only if cflag_!=0
                         bMax = cms.double(0.0) ## max impact param (fm); valid only if cflag_!=0
                         ))

generator.embeddingMode = 0

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/projects/CMSSW/rep/CMSSW/Configuration/Generator/python/Pyquen_GammaJet_pt20_2760GeV_cfi.py,v $'),
    annotation = cms.untracked.string('PYQUEN quenched gamma-jets (pt-hat > 20 GeV) at sqrt(s) = 2.76TeV')
    )

ProductionFilterSequence = cms.Sequence(generator)
