import FWCore.ParameterSet.Config as cms

## PFproducer for particle flow for PFCaloJets
#disable tracks and also input blocks are hltParticleFlowBlock

from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *


hltParticleFlowBlock = cms.EDProducer("PFBlockProducer",
    debug = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    elementImporters = cms.VPSet(
        cms.PSet(
            source = cms.InputTag("particleFlowClusterECAL"),
            importerName = cms.string('GenericClusterImporter')
        ),
        cms.PSet(
            source = cms.InputTag("particleFlowClusterHCAL"),
            importerName = cms.string('GenericClusterImporter')
        ),
        cms.PSet(
            source = cms.InputTag("particleFlowClusterHO"),
            importerName = cms.string('GenericClusterImporter')
        ),
        cms.PSet(
            source = cms.InputTag("particleFlowClusterHF"),
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


from RecoParticleFlow.PFProducer.particleFlow_cfi import particleFlowTmp

hltParticleFlow = particleFlowTmp.clone(
    GedPhotonValueMap = cms.InputTag(""),
    useEGammaFilters = cms.bool(False),
    useEGammaElectrons = cms.bool(False), 
    useEGammaSupercluster = cms.bool(False),
    rejectTracks_Step45 = cms.bool(False),
    usePFNuclearInteractions = cms.bool(False),  
    blocks = cms.InputTag("hltParticleFlowBlock"), 
    egammaElectrons = cms.InputTag(""),
    useVerticesForNeutral = cms.bool(False),
    PFEGammaCandidates = cms.InputTag(""),
    useProtectionsForJetMET = cms.bool(False),
    usePFConversions = cms.bool(False),
    rejectTracks_Bad = cms.bool(False),
    muons = cms.InputTag(""),
    postMuonCleaning = cms.bool(False),
    usePFSCEleCalib = cms.bool(False)
)


hltParticleFlowForJets = cms.Sequence( 
   particleFlowRecHitECAL*
   particleFlowRecHitHBHE*
   particleFlowRecHitHF*
   particleFlowRecHitHO*
   particleFlowClusterECALUncorrected*
   #particleFlowClusterECAL*
   particleFlowClusterHBHE*
   particleFlowClusterHCAL*
   particleFlowClusterHF*
   particleFlowClusterHO*
   hltParticleFlowBlock*
   hltParticleFlow
)


