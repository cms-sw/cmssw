import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
##################################################################
# Print the contents of the RawDataBuffer EDProduct (corresponding to DAQ raw data)
##################################################################
options = VarParsing.VarParsing ('analysis')

options.register ('inputDataSet',
                  'file:dth_dataset.root', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string, 
                  "Input ROOT dataset")

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(options.inputDataSet))
process.maxEvents.input = 1

process.dumpRawDataBuffer = cms.EDAnalyzer("DumpRawDataBuffer",
    minSLinkID = cms.uint32(0),
    maxSLinkID = cms.uint32(99999),
    #rawDataBufferTag = cms.InputTag("rawDataBufferProducer", "", "PROD"),
    rawDataBufferTag = cms.InputTag("rawDataCollector", "", "LHC"),
)

process.path = cms.Path(process.dumpRawDataBuffer)
