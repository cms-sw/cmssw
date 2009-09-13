# Auto generated configuration file
# using: 
# Revision: 1.123 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: alCaRecoSplitting_AlCaP0 -s ALCA:EcalCalPi0Calib+EcalCalEtaCalib+DQM --datatier ALCARECO --eventcontent ALCARECO --conditions FrontierConditions_GlobalTag,GR09_31X_V4P::All -n -1 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/EndOfProcess_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('step3_RELVAL nevts:-1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:ALCACombined.root')
)

# Additional output definition
process.ALCARECOStreamEcalCalPi0Calib = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPi0Calib:RECO')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEB_*', 
        'keep *_ecalPi0Corrected_pi0EcalRecHitsEE_*', 
        'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
        'keep *_hltAlCaPi0RegRecHits_pi0EcalRecHitsES_*',
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('ALCARECOEcalCalPi0Calib.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOEcalCalPi0Calib'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamEcalCalEtaCalib = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalEtaCalib:RECO')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ecalEtaCorrected_etaEcalRecHitsEB_*', 
        'keep *_ecalEtaCorrected_etaEcalRecHitsEE_*', 
        'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
        'keep *_hltAlCaEtaRegRecHits_etaEcalRecHitsES_*',
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('ALCARECOEcalCalEtaCalib.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOEcalCalEtaCalib'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Path and EndPath definitions
process.endjob_step = cms.Path(process.endOfProcess)
process.ALCARECOStreamEcalCalPi0CalibOutPath = cms.EndPath(process.ALCARECOStreamEcalCalPi0Calib)
process.ALCARECOStreamEcalCalEtaCalibOutPath = cms.EndPath(process.ALCARECOStreamEcalCalEtaCalib)

# Schedule definition
process.schedule = cms.Schedule(process.endjob_step,process.ALCARECOStreamEcalCalPi0CalibOutPath,process.ALCARECOStreamEcalCalEtaCalibOutPath)
