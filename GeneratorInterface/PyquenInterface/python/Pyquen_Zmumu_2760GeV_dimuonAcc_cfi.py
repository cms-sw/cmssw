import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PyquenDefaultSettings_cff import *
from GeneratorInterface.HiGenCommon.dileptonTrigSettings_cff import *

hiSignal = cms.EDFilter("PyquenGeneratorFilter",
                         dimuonAcceptance,        # dimuon acceptance filter
                         collisionParameters,
                         qgpParameters,
                         pyquenParameters,
                         doQuench = cms.bool(True),
                         bFixed = cms.double(0.0), ## fixed impact param (fm); valid only if cflag_=0
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     parameterSets = cms.vstring('pythiaUESettings',
                                                                                 'customProcesses',
                                                                                 'pythiaZsingle',
                                                                                 'pythiaZtoMuons',
                                                                                 'kinematics'),
                                                     kinematics = cms.vstring ("CKIN(7)=-2.",  #min rapidity
                                                                               "CKIN(8)=2."    #max rapidity
                                                                               ),
                                                     pythiaZsingle = cms.vstring ("MSUB(1)=1", # Z bosons
                                                                                  "MSTP(43)=2" # only Z (no gamma*)
                                                                                  ),
                                                     
                                                     ),
                        cFlag = cms.int32(0), ## centrality flag
                        bMin = cms.double(0.0), ## min impact param (fm); valid only if cflag_!=0
                        bMax = cms.double(0.0) ## max impact param (fm); valid only if cflag_!=0
                        )

hiSignal.embeddingMode = True
