import FWCore.ParameterSet.Config as cms

import FastSimulation.Muons.TSGFromL2_cfi as TSG

def SwitchToOIState(process):
    process.hltL3TrajectorySeedOIS = TSG.l3seeds("OIState")

def SwitchToOIHit(process):
    process.hltL3TrajectorySeedOIH = TSG.l3seeds("OIHit")

def SwitchToIOHit(process):
    process.hltL3TrajectorySeedIOH = TSG.l3seeds("IOHit")

def SwitchToOIHitCascade(process):
    process.hltL3TrajectorySeedOIHCascade = TSG.l3seeds("OIHitCascade")

def SwitchToCascade(process):
    # Make three individual seeders
    process.hltL3TrajectorySeedOIS = TSG.l3seeds("OIState")
    process.hltL3TrajectorySeedOIHC = TSG.l3seeds("OIHitCascade")
    process.hltL3TrajectorySeedIOHC = TSG.l3seeds("IOHitCascade")

    # Make one TrackCand for each seeder
    import FastSimulation.Muons.TrackCandidateFromL2_cfi
    process.hltL3TrackCandidateFromL2OIS = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
    process.hltL3TrackCandidateFromL2OIS.SeedProducer = "hltL3TrajectorySeedOIS"
    process.hltL3TrackCandidateFromL2OIHC = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
    process.hltL3TrackCandidateFromL2OIHC.SeedProducer = "hltL3TrajectorySeedOIHC"    
    process.hltL3TrackCandidateFromL2IOHC = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
    process.hltL3TrackCandidateFromL2IOHC.SeedProducer = "hltL3TrajectorySeedIOHC"

    # Make one Track for each TrackCand
    process.hltL3TkTracksFromL2OIS = process.hltL3TkTracksFromL2.clone()
    process.hltL3TkTracksFromL2OIS.src = "hltL3TrackCandidateFromL2OIS"
    process.hltL3TkTracksFromL2OIHC = process.hltL3TkTracksFromL2.clone()
    process.hltL3TkTracksFromL2OIHC.src = "hltL3TrackCandidateFromL2OIHC"
    process.hltL3TkTracksFromL2IOHC = process.hltL3TkTracksFromL2.clone()
    process.hltL3TkTracksFromL2IOHC.src = "hltL3TrackCandidateFromL2IOHC"

    # Make one L3Muons for each Track
    process.hltL3MuonsOIS = process.hltL3Muons.clone()
    process.hltL3MuonsOIS.L3TrajBuilderParameters.tkTrajLabel = "hltL3TkTracksFromL2OIS"
    process.hltL3MuonsOIHC = process.hltL3Muons.clone()
    process.hltL3MuonsOIHC.L3TrajBuilderParameters.tkTrajLabel = "hltL3TkTracksFromL2OIHC"
    process.hltL3MuonsIOHC = process.hltL3Muons.clone()
    process.hltL3MuonsIOHC.L3TrajBuilderParameters.tkTrajLabel = "hltL3TkTracksFromL2IOHC"

    process.hltL3MuonsOICombined = cms.EDProducer(
        "L3TrackCombiner",
        labels = cms.VInputTag(
           cms.InputTag("hltL3MuonsOIS"),
           cms.InputTag("hltL3MuonsOIHC"),
           )
        )

    process.l3MuonsAllCombined = cms.EDProducer(
        "L3TrackCombiner",
        labels = cms.VInputTag(
           cms.InputTag("hltL3MuonsOIS"),
           cms.InputTag("hltL3MuonsOIHC"),
           cms.InputTag("hltL3MuonsIOHC")
           )
        )

    process.l3TkFromL2Combination = cms.EDProducer(
        "L3TrackCombiner",
        labels = cms.VInputTag(
           cms.InputTag("hltL3TkTracksFromL2OIS"),
           cms.InputTag("hltL3TkTracksFromL2OIHC"),
           cms.InputTag("hltL3TkTracksFromL2IOHC")
           )
        )

    process.l3TkCandFromL2Combination = cms.EDProducer(
        "L3TrackCandCombiner",
        labels = cms.VInputTag(
           cms.InputTag("hltL3TrackCandidateFromL2OIS"),
           cms.InputTag("hltL3TrackCandidateFromL2OIHC"),
           cms.InputTag("hltL3TrackCandidateFromL2IOHC"),
           )
        )

    process.l3SeedCombination =  cms.EDProducer(
        "L3MuonTrajectorySeedCombiner",
        labels = cms.VInputTag(
           cms.InputTag("hltL3TrajectorySeedOIS"),
           cms.InputTag("hltL3TrajectorySeedOIHC"),
           cms.InputTag("hltL3TrajectorySeedIOHC")
           )
        )

    process.HLTL3muonTkCandidateSequenceOIS = cms.Sequence(
        process.HLTDoLocalPixelSequence +
        process.HLTDoLocalStripSequence +
        process.hltL3TrajectorySeedOIS +
        process.hltL3TrackCandidateFromL2OIS
        )
    
    process.HLTL3muonrecoNocandSequenceOIS = cms.Sequence(
        process.HLTL3muonTkCandidateSequenceOIS +
        process.hltL3TkTracksFromL2OIS +
        process.hltL3MuonsOIS
        )

    process.HLTL3muonTkCandidateSequenceOIHC = cms.Sequence(
        process.HLTL3muonrecoNocandSequenceOIS +
        process.hltL3TrajectorySeedOIHC +
        process.hltL3TrackCandidateFromL2OIHC
        )

    process.HLTL3muonrecoNocandSequenceOIHC = cms.Sequence(
        process.HLTL3muonTkCandidateSequenceOIHC +
        process.hltL3TkTracksFromL2OIHC +
        process.hltL3MuonsOIHC
        )

    process.HLTL3muonTkCandSequenceIOHC = cms.Sequence(
        process.HLTL3muonrecoNocandSequenceOIHC +
        process.hltL3MuonsOICombined +
        process.hltL3TrajectorySeedIOHC +
        process.hltL3TrackCandidateFromL2IOHC
        )

    process.HLTL3muonrecoNocandSequenceIOHC = cms.Sequence(
        process.HLTL3muonTkCandSequenceIOHC +
        process.hltL3TkTracksFromL2IOHC +
        process.hltL3MuonsIOHC
        )

    process.hltL3TrajectorySeed = process.l3SeedCombination
    process.hltL3TrackCandidateFromL2 = process.l3TkCandFromL2Combination
    process.hltL3TkTracksFromL2 = process.l3TkFromL2Combination
    process.hltL3Muons = process.l3MuonsAllCombined

    process.HLTL3muonTkCandidateSequence = cms.Sequence(
        process.HLTL3muonrecoNocandSequenceIOHC +
        process.hltL3TrajectorySeed +
        process.hltL3TrackCandidateFromL2
        )
    
##     process.HLTL3muonrecoNocandSequence = cms.Sequence(
##         process.HLTL3muonTkCandidateSequence +
##         process.hltL3TkTracksFromL2 +
##         process.hltL3Muons
##         )
