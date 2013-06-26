import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')
ivars.register('initialEvent',mult=ivars.multiplicity.singleton,info="for testing")

ivars.files = 'file:/mnt/hadoop/cms/store/user/yetkin/MC_Production/Pythia80_HydjetDrum_mix01/RECO/set2_random40000_HydjetDrum_642.root'

ivars.output = 'test.root'
ivars.maxEvents = -1
ivars.initialEvent = 1

ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process('TRACKATTACK')

doRegit=True
rawORreco=True
isEmbedded=True

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

#####################################################################################
# Input source
#####################################################################################

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            fileNames = cms.untracked.vstring(
    ivars.files
    ))

process.Timing = cms.Service("Timing")

# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(ivars.maxEvents))


#####################################################################################
# Load some general stuff
#####################################################################################

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff')
# Data Global Tag 44x 
#process.GlobalTag.globaltag = 'GR_P_V27::All'

# MC Global Tag 44x 
process.GlobalTag.globaltag = 'STARTHI44_V7::All'

# load centrality
from CmsHi.Analysis2010.CommonFunctions_cff import *
overrideCentrality(process)
process.HeavyIonGlobalParameters = cms.PSet(
	centralityVariable = cms.string("HFhits"),
	nonDefaultGlauberModel = cms.string("Hydjet_2760GeV"),
	centralitySrc = cms.InputTag("hiCentrality")
	)

process.hiCentrality.pixelBarrelOnly = False

#process.load("RecoHI.HiCentralityAlgos.CentralityFilter_cfi")
#process.centralityFilter.selectedBins = [0,1]

# EcalSeverityLevel ES Producer
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("RecoEcal.EgammaCoreTools.EcalNextToDeadChannelESProducer_cff")


#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string(ivars.output))

#####################################################################################
# Additional Reconstruction 
#####################################################################################


# redo reco or just tracking

if rawORreco:
    process.rechits = cms.Sequence(process.siPixelRecHits * process.siStripMatchedRecHits)
    process.hiTrackReco = cms.Sequence(process.rechits * process.heavyIonTracking)


    process.trackRecoAndSelection = cms.Path(
        #process.centralityFilter*
        process.hiTrackReco 
        )
    
else:
    process.reco_extra = cms.Path(
        #process.centralityFilter *
        process.RawToDigi * process.reconstructionHeavyIons)
    

    
# tack on iteative tracking, particle flow and calo-matching

#iteerative tracking
process.load("RecoHI.HiTracking.hiIterTracking_cff")
process.heavyIonTracking *= process.hiIterTracking


# Now do more tracking around the jets

if doRegit:
    process.load("RecoHI.HiTracking.hiRegitTracking_cff")
    
    process.hiRegitInitialStepSeeds.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
    process.hiRegitLowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
    process.hiRegitPixelPairStepSeeds.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
    process.hiRegitDetachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
    process.hiRegitMixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
    process.hiRegitMixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
    

    # merged with the global, iterative tracking
    process.load("RecoHI.HiTracking.MergeRegit_cff")


    
    # now re-run the muons
    process.regGlobalMuons = process.globalMuons.clone(
        TrackerCollectionLabel = "hiGeneralAndRegitTracks"
        )
    process.regGlbTrackQual = process.glbTrackQual.clone(
        InputCollection = "regGlobalMuons",
        InputLinksCollection = "regGlobalMuons"
        )
    process.regMuons = process.muons.clone()
    process.regMuons.TrackExtractorPSet.inputTrackCollection = "hiGeneralAndRegitTracks"
    process.regMuons.globalTrackQualityInputTag = "regGlbTrackQual"
    process.regMuons.inputCollectionLabels = cms.VInputTag("hiGeneralAndRegitTracks", "regGlobalMuons", "standAloneMuons:UpdatedAtVtx", "tevMuons:firstHit", "tevMuons:picky",
                                                           "tevMuons:dyt")
    
    
    process.regMuonReco = cms.Sequence(
        process.regGlobalMuons*
        process.regGlbTrackQual*
        process.regMuons
        )
    

    
    
    process.regionalTracking = cms.Path(
        process.hiRegitTracking *
        process.hiGeneralAndRegitTracks*
        process.regMuonReco 
        )
    
    
process.load("edwenger.HiTrkEffAnalyzer.HiTPCuts_cff")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("MitHig.PixelTrackletAnalyzer.trackAnalyzer_cff")
process.cutsTPForEff.primaryOnly = False
process.cutsTPForFak.ptMin = 0.2
process.cutsTPForEff.ptMin = 0.2

if doRegit:
    process.anaTrack.trackSrc = 'hiGeneralAndRegitTracks'
    process.anaTrack.qualityString = "highPurity"
else:
    process.anaTrack.trackSrc = 'hiGeneralTracks'
    process.anaTrack.qualityString = "highPurity"

process.anaTrack.trackPtMin = 0
process.anaTrack.useQuality = False
process.anaTrack.doPFMatching = False
process.anaTrack.doSimTrack = True    

process.trackAnalysis = cms.Path(
    process.cutsTPForEff*
    process.cutsTPForFak*
    process.anaTrack
    )


#####################################################################################
# Edm Output
#####################################################################################

#process.out = cms.OutputModule("PoolOutputModule",
#                               fileName = cms.untracked.string("/tmp/mnguyen/output.root")
#                               )
#process.save = cms.EndPath(process.out)
