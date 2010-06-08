# Auto generated configuration file
# using: 
# Revision: 1.172 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: alCaRecoSplitting -s ALCAOUTPUT:MuAlCalIsolatedMu+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+HcalCalHOCosmics+TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo --conditions FrontierConditions_GlobalTag,GR10_P_V5::All --no_ex ec --triggerResultsProcess RECO --filein=/store/temp/data/Run2010A/Mu/ALCARECO/v1/000/135/836/8C8EBAAF-F963-DF11-BB6A-000423D999 7E.root --no_output --python_filename=alCaRecoSplitting_Cosmics_cfg.py --data
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCAOUTPUT')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.172 $'),
    annotation = cms.untracked.string('alCaRecoSplitting nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(

)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:../promptReco_RAW2DIGI_L1Reco_RECO_DQM_ALCAPRODUCER_secondary.root')
)

# Additional output definition
process.ALCARECOStreamMuAlStandAloneCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlStandAloneCosmics_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlStandAloneCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('MuAlStandAloneCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlCosmics0T = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0T:RECO', 
            'pathALCARECOTkAlCosmicsCosmicTF0T:RECO')
    ),
    outputCommands = process.OutALCARECOTkAlCosmics0T_noDrop.outputCommands,
    fileName = cms.untracked.string('TkAlCosmics0T.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('TkAlCosmics0T'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlBeamHaloOverlaps = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlBeamHaloOverlaps:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlBeamHaloOverlaps_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlBeamHaloOverlaps.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('MuAlBeamHaloOverlaps'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamHcalCalHOCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHOCosmics:RECO')
    ),
    outputCommands = process.OutALCARECOHcalCalHOCosmics_noDrop.outputCommands,
    fileName = cms.untracked.string('HcalCalHOCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('HcalCalHOCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlBeamHalo = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlBeamHalo:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlBeamHalo_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlBeamHalo.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('MuAlBeamHalo'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlGlobalCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlGlobalCosmics:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlGlobalCosmics_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlGlobalCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('MuAlGlobalCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlCalIsolatedMu = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlCalIsolatedMu:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlCalIsolatedMu_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlCalIsolatedMu.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('MuAlCalIsolatedMu'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlBeamHalo = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlBeamHalo:RECO')
    ),
    outputCommands = process.OutALCARECOTkAlBeamHalo_noDrop.outputCommands,
    fileName = cms.untracked.string('TkAlBeamHalo.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('TkAlBeamHalo'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Other statements
process.GlobalTag.globaltag = 'GR10_P_V5::All'

# Path and EndPath definitions
process.pathALCARECOHcalCalHOCosmics = cms.Path(process.seqALCARECOHcalCalHOCosmics)
process.pathALCARECOMuAlStandAloneCosmics = cms.Path(process.seqALCARECOMuAlStandAloneCosmics*process.ALCARECOMuAlStandAloneCosmicsDQM)
process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
process.pathALCARECOTkAlCosmicsCTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
process.pathALCARECOTkAlCosmicsCTF = cms.Path(process.seqALCARECOTkAlCosmicsCTF*process.ALCARECOTkAlCosmicsCTFDQM)
process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
process.pathALCARECOTkAlCosmicsCTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCTFHLT*process.ALCARECOTkAlCosmicsCTFDQM)
process.pathALCARECODtCalib = cms.Path(process.seqALCARECODtCalib*process.ALCARECODTCalibSynchDQM)
process.pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFHLT*process.ALCARECOTkAlCosmicsCosmicTFDQM)
process.pathALCARECOHcalCalMinBias = cms.Path(process.seqALCARECOHcalCalMinBias*process.ALCARECOHcalCalPhisymDQM)
process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
process.pathALCARECOEcalCalPi0Calib = cms.Path(process.seqALCARECOEcalCalPi0Calib*process.ALCARECOEcalCalPi0CalibDQM)
process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
process.pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0T*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
process.pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCTF0THLT*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
process.pathALCARECOEcalCalEtaCalib = cms.Path(process.seqALCARECOEcalCalEtaCalib*process.ALCARECOEcalCalEtaCalibDQM)
process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias*process.ALCARECOSiStripCalMinBiasDQM)
process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
process.pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0THLT*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
process.pathALCARECOTkAlCosmicsCosmicTF = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF*process.ALCARECOTkAlCosmicsCosmicTFDQM)
process.pathALCARECOEcalCalPhiSym = cms.Path(process.seqALCARECOEcalCalPhiSym*process.ALCARECOEcalCalPhisymDQM)
process.pathALCARECOMuAlGlobalCosmics = cms.Path(process.seqALCARECOMuAlGlobalCosmics*process.ALCARECOMuAlGlobalCosmicsDQM)
process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)


process.ALCARECOStreamMuAlStandAloneCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlStandAloneCosmics)
process.ALCARECOStreamTkAlCosmics0TOutPath = cms.EndPath(process.ALCARECOStreamTkAlCosmics0T)
process.ALCARECOStreamMuAlBeamHaloOverlapsOutPath = cms.EndPath(process.ALCARECOStreamMuAlBeamHaloOverlaps)
process.ALCARECOStreamHcalCalHOCosmicsOutPath = cms.EndPath(process.ALCARECOStreamHcalCalHOCosmics)
process.ALCARECOStreamMuAlBeamHaloOutPath = cms.EndPath(process.ALCARECOStreamMuAlBeamHalo)
process.ALCARECOStreamMuAlGlobalCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlGlobalCosmics)
process.ALCARECOStreamMuAlCalIsolatedMuOutPath = cms.EndPath(process.ALCARECOStreamMuAlCalIsolatedMu)
process.ALCARECOStreamTkAlBeamHaloOutPath = cms.EndPath(process.ALCARECOStreamTkAlBeamHalo)

# Schedule definition
process.schedule = cms.Schedule(process.ALCARECOStreamMuAlStandAloneCosmicsOutPath,process.ALCARECOStreamTkAlCosmics0TOutPath,process.ALCARECOStreamMuAlBeamHaloOverlapsOutPath,process.ALCARECOStreamHcalCalHOCosmicsOutPath,process.ALCARECOStreamMuAlBeamHaloOutPath,process.ALCARECOStreamMuAlGlobalCosmicsOutPath,process.ALCARECOStreamMuAlCalIsolatedMuOutPath,process.ALCARECOStreamTkAlBeamHaloOutPath)
