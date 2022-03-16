import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.ProcessModifiers.trackingMkFit_cff import trackingMkFit

process = cms.Process('DUMP',Run3,trackingMkFit)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
# process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
# process.load('Configuration.EventContent.EventContent_cff')
# process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
# process.load('Configuration.StandardSequences.RawToDigi_cff')
# process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
# process.load('CommonTools.ParticleFlow.EITopPAG_cff')
# process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
# process.load('Configuration.StandardSequences.PATMC_cff')
# process.load('Configuration.StandardSequences.Validation_cff')
# process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
# process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


process.add_(cms.ESProducer("MkFitGeometryESProducer"))

defaultOutputFileName="phase1-trackerinfo.bin"

# level: 0 - no printout; 1 - print layers, 2 - print modules
# outputFileName: binary dump file; no dump if empty string
process.dump = cms.EDAnalyzer("DumpMkFitGeometry",
                              level   = cms.untracked.int32(2),
                              tagInfo = cms.untracked.string('no-tag'),
                       outputFileName = cms.untracked.string(defaultOutputFileName)
                              )

print("NOT YET IMPLEMENTED: Dumping geometry in ", defaultOutputFileName, "\n");
process.p = cms.Path(process.dump)
