# Auto generated configuration file
# using: 
# Revision: 1.372.2.1 
# Source: /local/reps/CMSSW.admin/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: reco --step RAW2DIGI,RECO --conditions GR_P_V32::All --eventcontent AOD --no_exec --data --filein file:outputPhysicsDST.root --fileout outputPhysicsDST_HLTplusAOD.root --python_filename promptReco_RAW2DIGI_AOD.py --number -1
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:outputPhysicsDST.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('reco nevts:-1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.AODoutput = cms.OutputModule("PoolOutputModule",
    eventAutoFlushCompressedSize = cms.untracked.int32(15728640),
    outputCommands = process.AODEventContent.outputCommands,
    fileName = cms.untracked.string('outputPhysicsDST_HLTplusAOD.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

process.AODoutput.outputCommands.extend(
        cms.untracked.vstring(
       'keep *_hltActivityPhotonClusterShape_*_*',
       'keep *_hltActivityPhotonEcalIso_*_*',
       'keep *_hltActivityPhotonHcalForHE_*_*',
       'keep *_hltActivityPhotonHcalIso_*_*',
       'keep *_hltCaloJetIDPassed_*_*',
       'keep *_hltElectronActivityDetaDphi_*_*',
       'keep *_hltHitElectronActivityTrackIsol_*_*',
       'keep *_hltKT6CaloJets_rho*_*',
       'keep *_hltL3MuonCandidates_*_*',
       'keep *_hltL3MuonCombRelIsolations_*_*',
       'keep *_hltMetClean_*_*',
       'keep *_hltMet_*_*',
       'keep *_hltPixelMatchElectronsActivity_*_*',
       'keep *_hltPixelVertices_*_*',
       'keep *_hltRecoEcalSuperClusterActivityCandidate_*_*',
       'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
       'keep edmTriggerResults_*_*_*',
       ## RAW data                                                                                                    
       #'keep FEDRawDataCollection_rawDataCollector_*_*',           
       #'keep FEDRawDataCollection_source_*_*', 
       'keep triggerTriggerEvent_*_*_*',
       'keep *_hltL1GtObjectMap_*_*'
            )
          )


# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'GR_P_V32::All'

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.AODoutput_step = cms.EndPath(process.AODoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.endjob_step,process.AODoutput_step)

