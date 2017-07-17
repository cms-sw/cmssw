import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')
ivars.register('initialEvent',mult=ivars.multiplicity.singleton,info="for testing")

ivars.files =     'file:/mnt/hadoop/cms/store/user/yetkin/MC_Production/Pythia80_HydjetDrum_mix01/RECO/set2_random40000_HydjetDrum_642.root'

ivars.output = 'test.root'
ivars.maxEvents = -1
ivars.initialEvent = 1

ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process('TRACKATTACK')


rawORreco=True
isEmbedded=True

process.Timing = cms.Service("Timing")

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
#process.centralityFilter.selectedBins = [0,1,2,3]

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

# Define Analysis sequencues
process.load('CmsHi.JetAnalysis.ExtraPfReco_cff')

############
# Extra RECO
############



# redo reco or just tracking

if rawORreco:
    process.rechits = cms.Sequence(process.siPixelRecHits * process.siStripMatchedRecHits)
    process.hiTrackReco = cms.Sequence(process.rechits * process.heavyIonTracking)

    
    process.reco_extra =  cms.Path(
        #process.centralityFilter
        process.hiTrackReco
        *process.muonRecoPbPb
        *process.electronGsfTrackingHi        
        *process.hiParticleFlowLocalReco
        *process.gsfEcalDrivenElectronSequence
        *process.hiParticleFlowReco
        *process.PFTowers
        )        
else:
    process.reco_extra = cms.Path(process.RawToDigi * process.reconstructionHeavyIons_withPF)
    
# tack on iteative tracking, particle flow and calo-matching

#iteerative tracking
process.load("RecoHI.HiTracking.hiIterTracking_cff")
process.heavyIonTracking *= process.hiIterTracking

# redo muons
process.globalMuons.TrackerCollectionLabel = "hiGeneralTracks"
process.muons.TrackExtractorPSet.inputTrackCollection = "hiGeneralTracks"
process.muons.inputCollectionLabels = ["hiGeneralTracks", "globalMuons", "standAloneMuons:UpdatedAtVtx", "tevMuons:firstHit", "tevMuons:picky", "tevMuons:dyt"]

# paricle flow
process.particleFlowClusterPS.thresh_Pt_Seed_Endcap = cms.double(99999.)
process.pfTrack.UseQuality = True # ! Should check that loose and tight tracks don't screw up PF
process.pfTrack.TrackQuality = cms.string('loose')
process.pfTrack.TkColList = cms.VInputTag("hiGeneralTracks")
#process.pfTrack.GsfTracksInEvents = cms.bool(False)

# do calo matching
process.load("RecoHI.HiTracking.HICaloCompatibleTracks_cff")
process.hiGeneralCaloMatchedTracks = process.hiCaloCompatibleTracks.clone(
    srcTracks = 'hiGeneralTracks'
    )

process.hiParticleFlowReco *= process.hiGeneralCaloMatchedTracks
    
process.hiCaloMatchFilteredTracks = cms.EDFilter("TrackSelector",
                                                 src = cms.InputTag("hiGeneralCaloMatchedTracks"),
                                                 cut = cms.string(
    'quality("highPuritySetWithPV")')                                                                                            
                                                 )

process.hiParticleFlowReco*=process.hiCaloMatchFilteredTracks



#  Track Analyzers
########################

process.load("edwenger.HiTrkEffAnalyzer.HiTPCuts_cff")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")

#process.cutsTPForEff.primaryOnly = False

process.cutsTPForFak.ptMin = 0.2
process.cutsTPForEff.ptMin = 0.2


process.load("MitHig.PixelTrackletAnalyzer.trackAnalyzer_cff")

process.anaTrack.trackSrc = 'hiGeneralCaloMatchedTracks'
process.anaTrack.qualityString = "highPuritySetWithPV"

process.anaTrack.trackPtMin = 0
process.anaTrack.useQuality = False
process.anaTrack.doPFMatching = False
process.anaTrack.doSimTrack = True
process.load("CmsHi.JetAnalysis.pfcandAnalyzer_cfi")
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0
process.interestingTrackEcalDetIds.TrackCollection = cms.InputTag("hiSelectedTracks")

process.anaTrack_hiSel = process.anaTrack.clone(trackSrc='hiSelectedTracks')
process.anaTrack_hiGen = process.anaTrack.clone(trackSrc='hiGeneralCaloMatchedTracks')

process.anaTrack_hiGen.doPFMatching = True
process.anaTrack_hiGen.pfCandSrc = 'particleFlowTmp'

process.anaTrack_hiSel.qualityString = "highPurity"
process.anaTrack_hiGen.qualityString = "highPurity"

process.load("edwenger.HiTrkEffAnalyzer.hitrkEffAnalyzer_cff")

process.hitrkEffAnalyzer_hiSel = process.hitrkEffAnalyzer.clone(tracks = 'hiSelectedTracks')
process.hitrkEffAnalyzer_hiGen = process.hitrkEffAnalyzer.clone(tracks = 'hiGeneralTracks')
process.hitrkEffAnalyzer_hiGenCalo = process.hitrkEffAnalyzer.clone(
    tracks = 'hiGeneralCaloMatchedTracks',
    qualityString = 'highPuritySetWithPV'
)
                          

process.trackAnalyzers = cms.Sequence(
    process.anaTrack*
    process.anaTrack_hiSel*                                      
    process.anaTrack_hiGen                                     
    )        

process.ana_step          = cms.Path(         #process.centralityFilter*
    process.cutsTPForEff*
    process.cutsTPForFak*
    process.trackAnalyzers
    )



#####################################################################################
# Edm Output
#####################################################################################

#process.out = cms.OutputModule("PoolOutputModule",
#                               fileName = cms.untracked.string("/tmp/mnguyen/output.root")
#                               )
#process.save = cms.EndPath(process.out)
