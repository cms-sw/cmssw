### Code : Make PFClusterjets, directly from #################
### PF clusters from ECAL and HCAL    ########################
### make PFCaloJets                ###########################
##############################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO2')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),                        
    fileNames = cms.untracked.vstring(
"/store/relval/CMSSW_7_4_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/MCRUN2_73_V9_postLS1beamspot-v1/00000/1AEA4813-169E-E411-BFBB-0025905A60E4.root"
     )
)


process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('PFL2_out.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)

process.RECOSIMoutput.outputCommands.append( "keep *_*_*_RECO2") 

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQMIO')
    )
)


process.load( "RecoJets.JetProducers.hltParticleFlowForJets_cfi" )
process.load( "RecoJets.JetProducers.ak4PFCaloJets_cfi" )

process.load( "RecoJets.JetProducers.PFClustersForJets_cff" )
process.load( "RecoJets.JetProducers.ak4PFClusterJets_cfi" )

process.PFClusterJet = cms.Path(
    process.pfClusterRefsForJets_step*
    process.ak4PFClusterJets
)

process.PFCaloJet = cms.Path( 
    process.hltParticleFlowForJets*
    process.ak4PFCaloJets
)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Path and EndPath definitions
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)


# Schedule definition
process.schedule = cms.Schedule(process.PFClusterJet, process.PFCaloJet, process.RECOSIMoutput_step)

# customisation of the proces
# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)
