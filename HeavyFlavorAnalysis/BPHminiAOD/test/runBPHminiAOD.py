from FWCore.ParameterSet.VarParsing import VarParsing
opt = VarParsing ('analysis')
opt.parseArguments()

outFileName = opt.outputFile
inFileNames = opt.inputFiles
Global_Tag  = 'auto:run2_mc'
MC          = True
Filter      = True

import FWCore.ParameterSet.Config as cms
process = cms.Process('PAT')

from HeavyFlavorAnalysis.BPHminiAOD.BPHminiAOD_cff import BPHminiAOD
BPHminiAOD(process,inFileNames,outFileName,Global_Tag,MC,Filter)
