import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PyquenDefaultSettings_cff import *

hiSignal = cms.EDFilter("PyquenGeneratorFilter",
                         collisionParameters,
                         qgpParameters,
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
                         )

hiSignal.embeddingMode = True

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Generator/python/Pyquen_GammaJet_pt20_2760GeV_cfi.py,v $'),
    annotation = cms.untracked.string('PYQUEN quenched gamma-jets (pt-hat > 20 GeV) at sqrt(s) = 2.76TeV')
    )

ProductionFilterSequence = cms.Sequence(hiSignal)
