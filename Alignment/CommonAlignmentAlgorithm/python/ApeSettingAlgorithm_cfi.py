# ApeSettingAlgorithm
# -------------------

import FWCore.ParameterSet.Config as cms

ApeSettingAlgorithm = cms.PSet(
    algoName = cms.string('ApeSettingAlgorithm'),
    saveApeToASCII = cms.untracked.bool(False),
    saveComposites = cms.untracked.bool(False),
    saveLocalNotGlobal = cms.untracked.bool(False),
    apeASCIISaveFile = cms.untracked.string('ApeDump.txt'),
    readApeFromASCII = cms.bool(False),
    readLocalNotGlobal = cms.bool(False),
    readFullLocalMatrix = cms.bool(False),
    setComposites = cms.bool(False),
    apeASCIIReadFile = cms.FileInPath('Alignment/CommonAlignmentAlgorithm/test/apeinput.txt')
)

# Parameters:
#    saveApeToASCII -- Do we write out an APE text file?
#    saveComposites -- Do we write APEs for composite detectors?
#    saveLocalNotGlobal -- Do we write the APEs in the local or global coordinates?
#    apeASCIISaveFile -- The name of the save-file.
#    readApeFromASCII -- Do we read in APEs from a text file?
#    readLocalNotGlobal -- Do we read APEs in the local or the global frame?
#    readFullLocalMatrix -- Do we read the full local matrix or just the diagonal elements?
# Full matrix format: DetID dxx dxy dyy dxz dyz dzz
# Diagonal element format: DetID sqrt(dxx) sqrt(dyy) sqrt(dzz)
#    setComposites -- Do we set the APEs for composite detectors or just ignore them?
#    apeASCIIReadFile -- Input file name.
# Also note:
#    process.AlignmentProducer.saveApeToDB -- to save as an sqlite file
# and associated entries in _cfg.py
