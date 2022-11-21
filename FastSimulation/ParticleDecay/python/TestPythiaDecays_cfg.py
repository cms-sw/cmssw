from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('analysis')
options.parseArguments()

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
                            fileNames = cms.untracked.vstring(options.inputFiles)
)
print(process.source.fileNames)

process.demo = cms.EDAnalyzer('TestPythiaDecays',
                              outputFile = cms.string(options.outputFile)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outputFile))

process.p = cms.Path(process.demo)
