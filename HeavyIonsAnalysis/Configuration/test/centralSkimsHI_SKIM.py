# Auto generated configuration file
# using: 
# Revision: 1.232.2.6 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: centralSkimsHI -s SKIM:DiJet+Photon+ZMM+ZEE --conditions GR10_P_V12::All --scenario HeavyIons --filein=/store/hidata/HIRun2010/HIAllPhysics/RECO/PromptReco-v1/000/150/063/B497BEDB-8BE8-DF11-B09D-0030487A18F2.root --data --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SkimsHeavyIons_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('centralSkimsHI nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/hidata/HIRun2010/HIAllPhysics/RECO/PromptReco-v1/000/150/063/B497BEDB-8BE8-DF11-B09D-0030487A18F2.root'),
    secondaryFileNames = cms.untracked.vstring("/store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/150/063/1C76D53B-88E8-DF11-B7D1-0030487CD812.root"),
    #eventsToProcess = cms.untracked.VEventRange('150063:517644-150063:517644'),
)

process.options = cms.untracked.PSet(
    #wantSummary = cms.untracked.bool(True)
)

#process.Timing = cms.Service("Timing")

# Output definition

# Additional output definition
process.SKIMStreamDiJet = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,                                           
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('diJetSkimPath')
    ),
    fileName = cms.untracked.string('DiJet.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('DiJet'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)
process.SKIMStreamPhoton = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('photonSkimPath')
    ),
    fileName = cms.untracked.string('Photon.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('Photon'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)
process.SKIMStreamZEE = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zEESkimPath')
    ),
    fileName = cms.untracked.string('ZEE.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('ZEE'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)
process.SKIMStreamZMM = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zMMSkimPath')
    ),
    fileName = cms.untracked.string('ZMM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('ZMM'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)

# Other statements
process.GlobalTag.globaltag = 'GR10_P_V12::All'

# Path and EndPath definitions
process.zEESkimPath = cms.Path(process.zEESkimSequence)

process.photonSkimPath = cms.Path(process.photonSkimSequence)

process.diJetSkimPath = cms.Path(process.diJetSkimSequence)

process.zMMSkimPath = cms.Path(process.zMMSkimSequence)

process.SKIMStreamDiJetOutPath = cms.EndPath(process.SKIMStreamDiJet)

process.SKIMStreamPhotonOutPath = cms.EndPath(process.SKIMStreamPhoton)

process.SKIMStreamZEEOutPath = cms.EndPath(process.SKIMStreamZEE)

process.SKIMStreamZMMOutPath = cms.EndPath(process.SKIMStreamZMM)


# Schedule definition
process.schedule = cms.Schedule(process.photonSkimPath,process.zMMSkimPath,process.zEESkimPath,process.diJetSkimPath,process.SKIMStreamDiJetOutPath,process.SKIMStreamPhotonOutPath,process.SKIMStreamZEEOutPath,process.SKIMStreamZMMOutPath)
