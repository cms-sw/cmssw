import FWCore.ParameterSet.Config as cms
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper as ticlDumper_

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_superclustering_dnn_cff import ticl_superclustering_dnn
from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, associatorsInstances


simTrackstersCollections = ["ticlSimTracksters", "ticlSimTrackstersfromCPs"]
dumperAssociators = []

for simTrackstersCollection in simTrackstersCollections:
    for tracksterIteration in ticlIterLabels:
        suffix = "CP" if "fromCPs" in simTrackstersCollection else "SC"
        dumperAssociators.append(
            cms.PSet(
                branchName=cms.string(tracksterIteration),
                suffix=cms.string(suffix),
                associatorRecoToSimInputTag=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{tracksterIteration}To{simTrackstersCollection}"),
                associatorSimToRecoInputTag=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{simTrackstersCollection}To{tracksterIteration}")
            )
        )


ticlDumper = ticlDumper_.clone(
    tracksterCollections = [*[cms.PSet(treeName=cms.string(label), inputTag=cms.InputTag(label)) for label in ticlIterLabels],
        cms.PSet(
            treeName=cms.string("simtrackstersSC"),
            inputTag=cms.InputTag("ticlSimTracksters"),
            tracksterType=cms.string("SimTracksterSC")
        ),
        cms.PSet(
            treeName=cms.string("simtrackstersCP"),
            inputTag=cms.InputTag("ticlSimTracksters", "fromCPs"),
            tracksterType=cms.string("SimTracksterCP")
        ),
    ],

    associators=dumperAssociators.copy(),
    saveSuperclustering = cms.bool(False)
)

ticl_v5.toModify(ticlDumper, 
                 ticlcandidates = cms.InputTag("ticlCandidate"), 
                 recoSuperClusters_sourceTracksterCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"), 
                 saveSuperclustering = cms.bool(True), 
                 trackstersInCand=cms.InputTag("ticlCandidate"))

(ticl_v5 & ticl_superclustering_mustache_pf).toModify(ticlDumper, saveSuperclustering=False, recoSuperClusters_sourceTracksterCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"))

simTrackstersBarrelCollections = ["ticlSimTrackstersBarrel", "ticlSimTrackstersBarrelfromCPs"]
ticlBarrelIterLabels = ["ticlTrackstersCLUE3DBarrel"]
dumperAssociatorsBarrel = []

for simTrackstersCollection in simTrackstersBarrelCollections:
    for tracksterIteration in ticlBarrelIterLabels:
        suffix = "CP" if "fromCPs" in simTrackstersCollection else "SC"
        dumperAssociatorsBarrel.append(
            cms.PSet(
                branchName=cms.string(tracksterIteration),
                suffix=cms.string(suffix),
                associatorRecoToSimInputTag=cms.InputTag(f"allBarrelTrackstersToSimTrackstersAssociationsByLCs:{tracksterIteration}To{simTrackstersCollection}"),
                associatorSimToRecoInputTag=cms.InputTag(f"allBarrelTrackstersToSimTrackstersAssociationsByLCs:{simTrackstersCollection}To{tracksterIteration}")
            )
        )

dumperAssociators+=dumperAssociatorsBarrel

ticl_barrel.toModify(ticlDumper,
                     tracksterCollections = [*[cms.PSet(treeName=cms.string(label), inputTag=cms.InputTag(label)) for label in ticlIterLabels+ticlBarrelIterLabels],
                        cms.PSet(                                                        
                            treeName=cms.string("simtrackstersSC"),
                            inputTag=cms.InputTag("ticlSimTracksters"),
                            tracksterType=cms.string("SimTracksterSC")
                        ),
                        cms.PSet(
                            treeName=cms.string("simtrackstersCP"),
                            inputTag=cms.InputTag("ticlSimTracksters", "fromCPs"),
                            tracksterType=cms.string("SimTracksterCP")
                        ),
                        cms.PSet(
                            treeName=cms.string("simtrackstersBarrelSC"),
                            inputTag=cms.InputTag("ticlSimTrackstersBarrel"),
                            tracksterType=cms.string("SimTracksterSC"),
                        ),
                        cms.PSet(
                            treeName=cms.string("simtrackstersBarrelCP"),
                            inputTag=cms.InputTag("ticlSimTrackstersBarrel", "fromCPs"),
                            tracksterType=cms.string("SimTracksterCP")
                        )
                     ],
                     associators=dumperAssociators.copy())
    
