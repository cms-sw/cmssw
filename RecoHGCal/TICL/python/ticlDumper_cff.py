import FWCore.ParameterSet.Config as cms
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper as ticlDumper_

from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet, associatorsInstances


dumperAssociators = []

for tracksterIteration in ticlIterLabelsPSet.labels:
    simTrackstersCollection = "ticlSimTrackstersfromCaloParticle"
    dumperAssociators.append(
        cms.PSet(
            branchName=cms.string(tracksterIteration),
            suffix=cms.string("CP"),
            associatorRecoToSimInputTag=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{tracksterIteration}To{simTrackstersCollection}"),
            associatorSimToRecoInputTag=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{simTrackstersCollection}To{tracksterIteration}")
        )
    )

    simTrackstersCollection = "ticlSimTrackstersfromBoundarySimCluster"
    dumperAssociators.append(
        cms.PSet(
            branchName=cms.string(tracksterIteration),
            suffix=cms.string("SC"),
            associatorRecoToSimInputTag=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{tracksterIteration}To{simTrackstersCollection}"),
            associatorSimToRecoInputTag=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{simTrackstersCollection}To{tracksterIteration}")
        )
    )


ticlDumper = ticlDumper_.clone(
    tracksterCollections = [*[cms.PSet(treeName=cms.string(label), inputTag=cms.InputTag(label)) for label in ticlIterLabelsPSet.labels],
        cms.PSet(
            treeName=cms.string("simtrackstersSC"),
            inputTag=cms.InputTag("ticlSimTracksters", "fromBoundarySimCluster"),
            tracksterType=cms.string("SimTracksterSC")
        ),
        cms.PSet(
            treeName=cms.string("simtrackstersCP"),
            inputTag=cms.InputTag("ticlSimTracksters", "fromCaloParticle"),
            tracksterType=cms.string("SimTracksterCP")
        ),
    ],

    associators=dumperAssociators.copy(),
    saveSuperclustering = cms.bool(True)
)


ticl_superclustering_mustache_pf.toModify(ticlDumper, saveSuperclustering=False, recoSuperClusters_sourceTracksterCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"))

simTrackstersBarrelCollections = ["ticlSimTrackstersBarrelfromBoundarySimCluster", "ticlSimTrackstersBarrelfromCaloParticle"]
ticlBarrelIterLabels = ["ticlTrackstersCLUE3DBarrel"]
dumperAssociatorsBarrel = []

for simTrackstersCollection in simTrackstersBarrelCollections:
    for tracksterIteration in ticlBarrelIterLabels:
        suffix = "CP" if "fromCaloParticle" in simTrackstersCollection else "SC"
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
                     tracksterCollections = [*[cms.PSet(treeName=cms.string(label), inputTag=cms.InputTag(label)) for label in ticlIterLabelsPSet.labels+ticlBarrelIterLabels],
                        cms.PSet(                                                        
                            treeName=cms.string("simtrackstersSC"),
                            inputTag=cms.InputTag("ticlSimTracksters", "fromBoundarySimCluster"),
                            tracksterType=cms.string("SimTracksterSC")
                        ),
                        cms.PSet(
                            treeName=cms.string("simtrackstersCP"),
                            inputTag=cms.InputTag("ticlSimTracksters", "fromCaloParticle"),
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
