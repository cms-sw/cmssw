#######################################################
###### implement the use of new ECal and HCal Linker###
###### which defined in                             ###
###### RecoParticleFlow/PFProducer/plugins/linkers/ ###
###### file name : ECALAndHCALCaloJetLinker.cc      ###
#######################################################


import FWCore.ParameterSet.Config as cms

hltParticleFlowBlock = cms.EDProducer(
    "PFBlockProducer",
    verbose = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),    
    elementImporters = cms.VPSet(        
        cms.PSet( importerName = cms.string("GeneralTracksImporter"),
                  source = cms.InputTag("hltLightPFTracks"),
                  muonSrc = cms.InputTag("hltMuons"),
                  useIterativeTracking = cms.bool(False),
                  DPtOverPtCuts_byTrackAlgo = cms.vdouble(0.5,0.5,0.5,0.5,0.5),
                  NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3)
                  ),        
        cms.PSet( importerName = cms.string("ECALClusterImporter"),
                  source = cms.InputTag("hltParticleFlowClusterECAL"),
                  BCtoPFCMap = cms.InputTag('') ),
        cms.PSet( importerName = cms.string("GenericClusterImporter"),
                  source = cms.InputTag("hltParticleFlowClusterHCAL") ),        
        cms.PSet( importerName = cms.string("GenericClusterImporter"),
                  source = cms.InputTag("hltParticleFlowClusterHFEM") ),
        cms.PSet( importerName = cms.string("GenericClusterImporter"),
                  source = cms.InputTag("hltParticleFlowClusterHFHAD") ),
        cms.PSet( importerName = cms.string("GenericClusterImporter"),
                  source = cms.InputTag("hltParticleFlowClusterPS") )        
        ),   
    linkDefinitions = cms.VPSet(
        cms.PSet( linkerName = cms.string("PreshowerAndECALLinker"),
                  linkType   = cms.string("PS1:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("PreshowerAndECALLinker"),
                  linkType   = cms.string("PS2:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TrackAndECALLinker"),
                  linkType   = cms.string("TRACK:ECAL"),
                  useKDTree  = cms.bool(True) ),
        cms.PSet( linkerName = cms.string("TrackAndHCALLinker"),
                  linkType   = cms.string("TRACK:HCAL"),
                  useKDTree  = cms.bool(True) ),
        #cms.PSet( linkerName = cms.string("ECALAndHCALLinker"),
        #          linkType   = cms.string("ECAL:HCAL"),
        #          useKDTree  = cms.bool(False) ),       
        cms.PSet( linkerName = cms.string("ECALAndHCALCaloJetLinker"), #New linker
                  linkType   = cms.string("ECAL:HCAL"),
                  useKDTree  = cms.bool(False) ),     
        cms.PSet( linkerName = cms.string("HFEMAndHFHADLinker"),
                  linkType   = cms.string("HFEM:HFHAD"),
                  useKDTree  = cms.bool(False) )
        )
) 

hltParticleFlowBlockPromptTracks = hltParticleFlowBlock.clone()
hltParticleFlowBlockPromptTracks.elementImporters[0].source = cms.InputTag("hltLightPFPromptTracks")

hltParticleFlowBlockForTaus = hltParticleFlowBlock.clone()
hltParticleFlowBlockForTaus.elementImporters[0].DPtOverPtCuts_byTrackAlgo = cms.vdouble(-1.0,-1.0,-1.0,-1.0,-1.0)

hltParticleFlowBlockReg = hltParticleFlowBlock.clone()
hltParticleFlowBlockReg.elementImporters[0].DPtOverPtCuts_byTrackAlgo = cms.vdouble(-1.0,-1.0,-1.0,-1.0,-1.0)
hltParticleFlowBlockReg.elementImporters[0].source = cms.InputTag("hltLightPFTracksReg")
