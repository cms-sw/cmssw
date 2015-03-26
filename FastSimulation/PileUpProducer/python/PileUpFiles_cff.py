import FWCore.ParameterSet.Config as cms

puFileNames_7TeV = cms.untracked.vstring(
    'MinBias7TeV_001.root', 
    'MinBias7TeV_002.root', 
    'MinBias7TeV_003.root', 
    'MinBias7TeV_004.root', 
    'MinBias7TeV_005.root', 
    'MinBias7TeV_006.root', 
    'MinBias7TeV_007.root', 
    'MinBias7TeV_008.root', 
    'MinBias7TeV_009.root', 
    'MinBias7TeV_010.root'
    )

puFileNames_8TeV = cms.untracked.vstring(
    'MinBias8TeV_001.root', 
    'MinBias8TeV_002.root', 
    'MinBias8TeV_003.root', 
    'MinBias8TeV_004.root', 
    'MinBias8TeV_005.root', 
    'MinBias8TeV_006.root', 
    'MinBias8TeV_007.root', 
    'MinBias8TeV_008.root', 
    'MinBias8TeV_009.root', 
    'MinBias8TeV_010.root'
    )

puFileNames_10TeV = cms.untracked.vstring(
    'MinBias10TeV_001.root', 
    'MinBias10TeV_002.root', 
    'MinBias10TeV_003.root', 
    'MinBias10TeV_004.root', 
    'MinBias10TeV_005.root', 
    'MinBias10TeV_006.root', 
    'MinBias10TeV_007.root', 
    'MinBias10TeV_008.root', 
    'MinBias10TeV_009.root', 
    'MinBias10TeV_010.root'
    )

puFileNames_13TeV = cms.untracked.vstring(
    'MinBias13TeV_001.root', 
    'MinBias13TeV_002.root', 
    'MinBias13TeV_003.root', 
    'MinBias13TeV_004.root', 
    'MinBias13TeV_005.root', 
    'MinBias13TeV_006.root', 
    'MinBias13TeV_007.root', 
    'MinBias13TeV_008.root', 
    'MinBias13TeV_009.root', 
    'MinBias13TeV_010.root'
    )

puFileNames_14TeV = cms.untracked.vstring(
    'MinBias14TeV_001.root', 
    'MinBias14TeV_002.root', 
    'MinBias14TeV_003.root', 
    'MinBias14TeV_004.root', 
    'MinBias14TeV_005.root', 
    'MinBias14TeV_006.root', 
    'MinBias14TeV_007.root', 
    'MinBias14TeV_008.root', 
    'MinBias14TeV_009.root', 
    'MinBias14TeV_010.root'
    )

puFileNames = cms.PSet(fileNames = puFileNames_8TeV)

#
# Modify for running in Run 2
#
from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify(puFileNames,fileNames=puFileNames_13TeV)

