import FWCore.ParameterSet.Config as cms

## particle Flow block producer for "PFCaloJets"
## PFCaloJets :  PFjets without tracks

hltParticleFlowBlock = cms.EDProducer("PFBlockProducer",
    debug = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    elementImporters = cms.VPSet(
        cms.PSet(
            source = cms.InputTag("hltParticleFlowClusterECAL"),
            importerName = cms.string('GenericClusterImporter')
        ), 
        cms.PSet(
            source = cms.InputTag("hltParticleFlowClusterHCAL"),
            importerName = cms.string('GenericClusterImporter')
        ), 
        cms.PSet(
            source = cms.InputTag("hltParticleFlowClusterHO"),
            importerName = cms.string('GenericClusterImporter')
        ), 
        cms.PSet(
            source = cms.InputTag("hltParticleFlowClusterHF"),
            importerName = cms.string('GenericClusterImporter')
        ) 
    ),
    linkDefinitions = cms.VPSet(
        cms.PSet(
            linkType = cms.string('ECAL:HCAL'),
            useKDTree = cms.bool(False),
            #linkerName = cms.string('ECALAndHCALLinker')
            linkerName = cms.string('ECALAndHCALCaloJetLinker') #new ECal and HCal Linker for PFCaloJets
        ), 
        cms.PSet(
            linkType = cms.string('HCAL:HO'),
            useKDTree = cms.bool(False),
            linkerName = cms.string('HCALAndHOLinker')
        ), 
        cms.PSet(
            linkType = cms.string('HFEM:HFHAD'),
            useKDTree = cms.bool(False),
            linkerName = cms.string('HFEMAndHFHADLinker')
        ), 
        cms.PSet(
            linkType = cms.string('ECAL:ECAL'),
            useKDTree = cms.bool(False),
            linkerName = cms.string('ECALAndECALLinker')
        ) 
   )
)

