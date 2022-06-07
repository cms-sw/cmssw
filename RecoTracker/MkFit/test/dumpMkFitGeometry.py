import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.ProcessModifiers.trackingMkFit_cff import trackingMkFit

process = cms.Process('DUMP',Run3,trackingMkFit)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.MkFitGeometryESProducer = dict(limit=-1)

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1


process.add_(cms.ESProducer("MkFitGeometryESProducer"))

defaultOutputFileName="phase1-trackerinfo.bin"

# level: 0 - no printout; 1 - print layers, 2 - print modules
# outputFileName: binary dump file; no dump if empty string
process.dump = cms.EDAnalyzer("DumpMkFitGeometry",
                              level   = cms.untracked.int32(1),
                       outputFileName = cms.untracked.string(defaultOutputFileName)
                              )

print("Requesting MkFit geometry dump into file:", defaultOutputFileName, "\n");
process.p = cms.Path(process.dump)
