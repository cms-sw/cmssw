# Harvesting of the BTL local reco validation (BtlLocalRecoHarvester) for the BTL test beam 2025 geometry with
# active DUT and REF modules. Reads the DQMIO output of BTLTB2025_step3_cfg.py and writes the DQM_V0001_*.root file.
#   cmsRun BTLTB2025_step4_cfg.py inputFiles=file:BTLTB2025_step3_inDQM.root
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.inputFiles = 'file:BTLTB2025_step3_inDQM.root'
options.parseArguments()

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
process = cms.Process('HARVESTING', Phase2C22I13M9, dd4hep)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source("DQMRootSource", fileNames = cms.untracked.vstring(options.inputFiles))

from Validation.MtdValidation.btlLocalRecoPostProcessor_cfi import btlLocalRecoPostProcessor
process.btlLocalRecoPostProcessor = btlLocalRecoPostProcessor

process.dqmSaver.workflow = '/BTLTB2025/beamAlongZ/VALID'

process.harvesting_step = cms.Path(process.btlLocalRecoPostProcessor)
process.dqmsave_step = cms.Path(process.DQMSaver)
process.schedule = cms.Schedule(process.harvesting_step, process.dqmsave_step)
