import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PyquenDefaultSettings_cff import *

generator = cms.EDFilter("PyquenGeneratorFilter",
                         collisionParameters,
                         qgpParameters,
                         pyquenParameters,
                         doQuench = cms.bool(True),
                         bFixed = cms.double(0.0), ## fixed impact param (fm); valid only if cflag_=0
                         PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                     parameterSets = cms.vstring('pythiaUESettings','customProcesses','pythiaZjets','pythiaZtoMuonsAndElectrons','kinematics'),
                                                     kinematics = cms.vstring ("CKIN(3)=10",  #min pthat
                                                                               "CKIN(4)=9999", #max pthat
                                                                               "CKIN(7)=-2.",  #min rapidity
                                                                               "CKIN(8)=2."    #max rapidity
                                                                               )
                                                     
                                                     ),
                        cFlag = cms.int32(0), ## centrality flag
                        bMin = cms.double(0.0), ## min impact param (fm); valid only if cflag_!=0
                        bMax = cms.double(0.0) ## max impact param (fm); valid only if cflag_!=0
                        )

generator.embeddingMode = False

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/projects/CMSSW/rep/CMSSW/Configuration/Generator/python/Pyquen_ZeemumuJets_pt10_2760GeV_cfi.py,v $'),
    annotation = cms.untracked.string('PYQUEN Z->mumu and Z->ee (pt-hat > 10 GeV) at sqrt(s) = 2.76TeV')
    )

ProductionFilterSequence = cms.Sequence(generator)
