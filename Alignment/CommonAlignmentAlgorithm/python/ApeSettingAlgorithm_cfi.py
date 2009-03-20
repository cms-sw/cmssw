# ApeSettingAlgorithm
# -------------------

import FWCore.ParameterSet.Config as cms

ApeSettingAlgorithm = cms.PSet(
    algoName = cms.string('ApeSettingAlgorithm'),
    saveApeToASCII = cms.untracked.bool(False),
    saveComposites = cms.untracked.bool(False),
    apeASCIISaveFile = cms.untracked.string('ApeDump.txt'),
    readApeFromASCII = cms.bool(False),
    readLocalNotGlobal = cms.bool(False),
    setComposites = cms.bool(False),
    apeASCIIReadFile = cms.FileInPath('Alignment/CommonAlignmentAlgorithm/test/apeinput.txt')
    #
    # add your parameters here
)

