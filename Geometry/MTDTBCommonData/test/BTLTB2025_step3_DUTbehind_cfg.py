# MTD local reco validation (BtlLocalRecoValidation) for the BTL test beam 2025 geometry with active DUT and REF
# modules (DD4hep). Runs on the output of BTLTB2025_step2_cfg.py (local reco already done there) and writes DQMIO.
# The SimHit crossing frames are rebuilt here by the mixing module in playback mode.
#   cmsRun BTLTB2025_step3_cfg.py inputFiles=file:BTLTB2025_step2.root outputFile=BTLTB2025_step3_inDQM.root
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.inputFiles = 'file:BTLTB2025_step2.root'
options.outputFile = 'BTLTB2025_step3_inDQM.root'
options.parseArguments()

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
process = cms.Process('VALID', Phase2C22I13M9, dd4hep)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Geometry.MTDTBCommonData.GeometryDD4hepExtendedBTLTB2025Reco_DUTbehind_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# ES producers of the MTD local reco (cluster parameter estimator MTDCPEBase, time calibration)
process.load('RecoLocalFastTime.Configuration.RecoLocalFastTime_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(options.inputFiles))
process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(1),
                                     numberOfStreams = cms.untracked.uint32(0))

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_0T', '')

# mixing module in playback mode, without digitizers, to rebuild the SimHit crossing frames used by the
# validation (as done by cmsDriver in the VALIDATION step). No ETL in the geometry: no FastTimerHitsEndcap to mix.
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in list(process.aliases_().keys()):
    delattr(process, a)
_mixSH = process.mix.mixObjects.mixSH
_mixSH.input = cms.VInputTag(*[t for t in _mixSH.input if t.productInstanceLabel != "FastTimerHitsEndcap"])
_mixSH.subdets = cms.vstring(*[x for x in _mixSH.subdets if x != "FastTimerHitsEndcap"])
_mixSH.crossingFrames = cms.untracked.vstring(*[x for x in _mixSH.crossingFrames if x != "FastTimerHitsEndcap"])

# reco cluster -> MtdSimLayerCluster association, as in Validation/Configuration/python/mtdSimValid_cff.py
from SimFastTiming.MtdAssociatorProducers.mtdRecoClusterToSimLayerClusterAssociatorByHits_cfi import mtdRecoClusterToSimLayerClusterAssociatorByHits
from SimFastTiming.MtdAssociatorProducers.mtdRecoClusterToSimLayerClusterAssociation_cfi import mtdRecoClusterToSimLayerClusterAssociation
process.mtdRecoClusterToSimLayerClusterAssociatorByHits = mtdRecoClusterToSimLayerClusterAssociatorByHits
process.mtdRecoClusterToSimLayerClusterAssociation = mtdRecoClusterToSimLayerClusterAssociation

from Validation.MtdValidation.btlLocalRecoValid_cfi import btlLocalRecoValid
process.btlLocalRecoValid = btlLocalRecoValid

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('DQMIO'), filterName = cms.untracked.string('')),
    fileName = cms.untracked.string('file:' + options.outputFile),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# MTDTrackingRecHits are rebuilt here: their pointer to the GeomDet is transient and is not restored when they
# are read back from the step2 file (globalPositionError() would dereference a null pointer in the validation)
process.validation_step = cms.Path(process.mix +
                                   process.mtdTrackingRecHits +
                                   process.mtdRecoClusterToSimLayerClusterAssociatorByHits +
                                   process.mtdRecoClusterToSimLayerClusterAssociation +
                                   process.btlLocalRecoValid)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
process.schedule = cms.Schedule(process.validation_step, process.DQMoutput_step)

process.MessageLogger.cerr.FwkReport.reportEvery = 500
