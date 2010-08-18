import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
# L2 seeds from L1 input
# module hltL2MuonSeeds = L2MuonSeeds from "RecoMuon/L2MuonSeedGenerator/data/L2MuonSeeds.cfi"
# replace hltL2MuonSeeds.GMTReadoutCollection = l1extraParticles
# replace hltL2MuonSeeds.InputObjects = l1extraParticles
# L3 regional reconstruction
from FastSimulation.Muons.L3Muons_cff import *
import FastSimulation.Muons.L3Muons_cfi
hltL3Muons = FastSimulation.Muons.L3Muons_cfi.L3Muons.clone()
hltL3Muons.L3TrajBuilderParameters.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
hltL3Muons.L3TrajBuilderParameters.TrackerRecHitBuilder = 'WithoutRefit'
hltL3Muons.TrackLoaderParameters.beamSpot = cms.InputTag("offlineBeamSpot")

# L3 regional seeding, candidating, tracking
#--the two below have to be picked up from confDB: 
# from FastSimulation.Muons.TSGFromL2_cfi import *
# from FastSimulation.Muons.HLTL3TkMuons_cfi import *
#from FastSimulation.Muons.TrackCandidateFromL2_cfi import *
#from FastSimulation.Muons.TSGFromL2_cfi import *
#hltL3TrajectorySeed = FastSimulation.Muons.TSGFromL2_cfi.hltL3TrajectorySeed.clone()

import FastSimulation.Muons.TSGFromL2_cfi as TSG
from FastSimulation.Muons.TSGFromL2_cfi import OIStatePropagators as OIProp
from FastSimulation.Muons.TSGFromL2_cfi import OIHitPropagators as OIHProp
# Make three individual seeders
hltL3TrajectorySeedOIS = TSG.l3seeds("OIState")
hltL3TrajectorySeedOIS.ServiceParameters.Propagators = cms.untracked.vstring()
OIProp(hltL3TrajectorySeedOIS,hltL3TrajectorySeedOIS.TkSeedGenerator)
hltL3TrajectorySeedOIHC = TSG.l3seeds("OIHitCascade")
hltL3TrajectorySeedOIHC.ServiceParameters.Propagators = cms.untracked.vstring()
OIHProp(hltL3TrajectorySeedOIHC,hltL3TrajectorySeedOIHC.TkSeedGenerator.iterativeTSG)
hltL3TrajectorySeedIOHC = TSG.l3seeds("IOHitCascade")

# Make one TrackCand for each seeder
from FastSimulation.Muons.TrackCandidateFromL2_cfi import *
hltL3TrackCandidateFromL2OIS = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2OIS.SeedProducer = "hltL3TrajectorySeedOIS"
hltL3TrackCandidateFromL2OIHC = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2OIHC.SeedProducer = "hltL3TrajectorySeedOIHC"    
hltL3TrackCandidateFromL2IOHC = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2IOHC.SeedProducer = "hltL3TrajectorySeedIOHC"

# Make one Track for each TrackCand
import FastSimulation.Muons.L3TkTracksFromL2_cfi
hltL3TkTracksFromL2OIS = FastSimulation.Muons.L3TkTracksFromL2_cfi.hltL3TkTracksFromL2.clone()
hltL3TkTracksFromL2OIS.src = "hltL3TrackCandidateFromL2OIS"
hltL3TkTracksFromL2OIHC = FastSimulation.Muons.L3TkTracksFromL2_cfi.hltL3TkTracksFromL2.clone()
hltL3TkTracksFromL2OIHC.src = "hltL3TrackCandidateFromL2OIHC"
hltL3TkTracksFromL2IOHC = FastSimulation.Muons.L3TkTracksFromL2_cfi.hltL3TkTracksFromL2.clone()
hltL3TkTracksFromL2IOHC.src = "hltL3TrackCandidateFromL2IOHC"

# Make one L3Muons for each Track
hltL3MuonsOIS = hltL3Muons.clone()
hltL3MuonsOIS.L3TrajBuilderParameters.tkTrajLabel = "hltL3TkTracksFromL2OIS"
hltL3MuonsOIHC = hltL3Muons.clone()
hltL3MuonsOIHC.L3TrajBuilderParameters.tkTrajLabel = "hltL3TkTracksFromL2OIHC"
hltL3MuonsIOHC = hltL3Muons.clone()
hltL3MuonsIOHC.L3TrajBuilderParameters.tkTrajLabel = "hltL3TkTracksFromL2IOHC"

hltL3MuonsOICombined = cms.EDProducer(
    "L3TrackCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3MuonsOIS"),
    cms.InputTag("hltL3MuonsOIHC"),
    )
    )

#l3MuonsAllCombined = cms.EDProducer(
hltL3Muons = cms.EDProducer(
    "L3TrackCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3MuonsOIS"),
    cms.InputTag("hltL3MuonsOIHC"),
    cms.InputTag("hltL3MuonsIOHC")
    )
    )

#l3TkFromL2Combination = cms.EDProducer(
hltL3TkTracksFromL2 = cms.EDProducer(
    "L3TrackCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3TkTracksFromL2OIS"),
    cms.InputTag("hltL3TkTracksFromL2OIHC"),
    cms.InputTag("hltL3TkTracksFromL2IOHC")
    )
    )

#l3TkCandFromL2Combination = cms.EDProducer(
hltL3TrackCandidateFromL2 = cms.EDProducer(
    "L3TrackCandCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3TrackCandidateFromL2OIS"),
    cms.InputTag("hltL3TrackCandidateFromL2OIHC"),
    cms.InputTag("hltL3TrackCandidateFromL2IOHC"),
    )
    )

#l3SeedCombination =  cms.EDProducer(
hltL3TrajectorySeed =  cms.EDProducer(
    "L3MuonTrajectorySeedCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3TrajectorySeedOIS"),
    cms.InputTag("hltL3TrajectorySeedOIHC"),
    cms.InputTag("hltL3TrajectorySeedIOHC")
    )
    )

HLTL3muonTkCandidateSequenceOIS = cms.Sequence(
    cms.SequencePlaceholder("HLTDoLocalPixelSequence") +
    cms.SequencePlaceholder("HLTDoLocalStripSequence") +
    hltL3TrajectorySeedOIS +
    hltL3TrackCandidateFromL2OIS
    )

HLTL3muonrecoNocandSequenceOIS = cms.Sequence(
    HLTL3muonTkCandidateSequenceOIS +
    hltL3TkTracksFromL2OIS +
    hltL3MuonsOIS
    )

HLTL3muonTkCandidateSequenceOIHC = cms.Sequence(
    HLTL3muonrecoNocandSequenceOIS +
    hltL3TrajectorySeedOIHC +
    hltL3TrackCandidateFromL2OIHC
    )

HLTL3muonrecoNocandSequenceOIHC = cms.Sequence(
    HLTL3muonTkCandidateSequenceOIHC +
    hltL3TkTracksFromL2OIHC +
    hltL3MuonsOIHC
    )

HLTL3muonTkCandSequenceIOHC = cms.Sequence(
    HLTL3muonrecoNocandSequenceOIHC +
    hltL3MuonsOICombined +
    hltL3TrajectorySeedIOHC +
    hltL3TrackCandidateFromL2IOHC
    )

HLTL3muonrecoNocandSequenceIOHC = cms.Sequence(
    HLTL3muonTkCandSequenceIOHC +
    hltL3TkTracksFromL2IOHC +
    hltL3MuonsIOHC
    )

#hltL3TrajectorySeed = l3SeedCombination
#hltL3TrackCandidateFromL2 = l3TkCandFromL2Combination
#hltL3TkTracksFromL2 = l3TkFromL2Combination
#hltL3Muons = l3MuonsAllCombined

HLTL3muonTkCandidateSequence = cms.Sequence(
    HLTL3muonrecoNocandSequenceIOHC +
    hltL3TrajectorySeed +
    hltL3TrackCandidateFromL2
    )


