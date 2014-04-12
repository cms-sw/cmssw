import FWCore.ParameterSet.Config as cms

process = cms.Process("SLINKTORAW")
process.load("IORawData.SiPixelInputSources.PixelSLinkDataInputSource_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myfile.root')
)

process.e = cms.EndPath(process.out)
process.PixelSLinkDataInputSource.fileNames = ['rfio:/castor/cern.ch/cms/store/TAC/PIXEL/FPIX/HC+Z1/SCurve_565.dmp']


