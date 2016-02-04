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
