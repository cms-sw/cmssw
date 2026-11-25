# DIGI (+ MTD local reco) for the BTL test beam 2025 geometry with active DUT and REF modules (DD4hep),
# adapted from BTLTB2025_step2_DUTbehind_cfg.py (BTL digitizer with the standard parameters, as in 2025)
#   cmsRun BTLTB2025_step2_cfg.py inputFiles=file:BTLTB2025_step1.root outputFile=BTLTB2025_step2.root
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.inputFiles = 'file:BTLTB2025_step1.root'
options.outputFile = 'BTLTB2025_step2.root'
options.register('localReco', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool,
                 "run the MTD local reconstruction")
options.parseArguments()

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
process = cms.Process('DIGI', Phase2C22I13M9, dd4hep)

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Geometry.MTDTBCommonData.GeometryDD4hepExtendedBTLTB2025Reco_DUTbehind_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('RecoLocalFastTime.Configuration.RecoLocalFastTime_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(options.inputFiles))
process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(1),
                                     numberOfStreams = cms.untracked.uint32(0))

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO'), filterName = cms.untracked.string('')),
    fileName = cms.untracked.string('file:' + options.outputFile),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_0T', '')

# keep only the MTD digitizer and the aliases of its products (as in the BTL test beam step2)
for name in list(process.aliases_().keys()):
    if name not in ("FTLBarrel", "FTLEndcap"):
        delattr(process, name)
for k in list(process.mix.digitizers.parameters_().keys()):
    if k not in ("fastTimingLayer", "MC_fastTimingLayer"):
        delattr(process.mix.digitizers, k)

# no ETL in the test beam geometry: there are no FastTimerHitsEndcap SimHits to be mixed
_mixSH = process.mix.mixObjects.mixSH
_mixSH.input = cms.VInputTag(*[t for t in _mixSH.input if t.productInstanceLabel != "FastTimerHitsEndcap"])
_mixSH.subdets = cms.vstring(*[x for x in _mixSH.subdets if x != "FastTimerHitsEndcap"])
_mixSH.crossingFrames = cms.untracked.vstring(*[x for x in _mixSH.crossingFrames if x != "FastTimerHitsEndcap"])
_mtdTruth = process.mix.digitizers.MC_fastTimingLayer
_mtdTruth.simHitCollections.mtdCollections = cms.VInputTag(cms.InputTag("g4SimHits", "FastTimerHitsBarrel"))

process.digitisation_step = cms.Path(cms.SequencePlaceholder("randomEngineStateProducer") * process.mix)
process.mtdlocalreco_step = cms.Path(process.fastTimingLocalReco)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)
steps = [process.digitisation_step]
if options.localReco:
    steps.append(process.mtdlocalreco_step)
process.schedule = cms.Schedule(*steps, process.endjob_step, process.FEVTDEBUGoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

process.MessageLogger.cerr.FwkReport.reportEvery = 50

from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
