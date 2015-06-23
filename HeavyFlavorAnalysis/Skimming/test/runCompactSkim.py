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

from HeavyFlavorAnalysis.Skimming.CompactSkim_cff import CompactSkim
CompactSkim(process,inFileNames,outFileName,Global_Tag,MC,Filter)
