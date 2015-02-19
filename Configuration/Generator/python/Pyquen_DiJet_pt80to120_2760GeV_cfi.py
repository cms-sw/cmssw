import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PyquenDefaultSettings_cff import *

generator = cms.EDFilter("PyquenGeneratorFilter",
                        collisionParameters,
                        qgpParameters,
                        pyquenParameters,
                        doQuench = cms.bool(True),
                        bFixed = cms.double(0.0), ## fixed impact param (fm); valid only if cflag_=0
                        PythiaParameters = cms.PSet(pyquenPythiaDefaultBlock,
                                                    parameterSets = cms.vstring('pythiaUESettings','ppJets','kinematics'),
                                                    kinematics = cms.vstring ("CKIN(3)=80",  #min pthat
                                                                              "CKIN(4)=120" #max pthat
                                                                              )
                                                    ),
                        cFlag = cms.int32(0), ## centrality flag
                        bMin = cms.double(0.0), ## min impact param (fm); valid only if cflag_!=0
                        bMax = cms.double(0.0) ## max impact param (fm); valid only if cflag_!=0
                        )

generator.embeddingMode = False

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /local/projects/CMSSW/rep/CMSSW/Configuration/Generator/python/Pyquen_DiJet_pt80to120_2760GeV_cfi.py,v $'),
    annotation = cms.untracked.string('PYQUEN quenched dijets (80 < pt-hat < 120 GeV) at sqrt(s) = 2.76TeV')
    )

ProductionFilterSequence = cms.Sequence(generator)
