import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixObjects_cfi import *
eventEmbeddingSourceParameters = cms.PSet(
    nbPileupEvents = cms.PSet(
        averageNumber = cms.double(1.0)
    ),
    seed = cms.int32(325),
    type = cms.string('fixed')
)
eventEmbeddingMixParameters = cms.PSet(
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0),
    Label = cms.string(''),
    bunchspace = cms.int32(125),
    checktof = cms.untracked.bool(False)
)
simEventEmbeddingMixParameters = cms.PSet(
    eventEmbeddingMixParameters,
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)
genEventEmbeddingMixParameters = cms.PSet(
    eventEmbeddingMixParameters,
    mixObjects = cms.PSet(
        mySet = cms.PSet(
            input = cms.VInputTag(cms.InputTag("source"), cms.InputTag("secsource")),
            type = cms.string('HepMCProduct')
        )
    )
)

mixSim = cms.EDProducer("MixingModule",
                        simEventEmbeddingMixParameters,
                        input = cms.SecSource("PoolRASource",
                                              eventEmbeddingSourceParameters,
                                              fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/hydjet_sim_x2_c10_d20080425/hydjet_sim_x2_c10_d20080425_r000002.root')
                                              )
                             )


mixGen = cms.EDProducer("MixingModule",
                        genEventEmbeddingMixParameters,
                        input = cms.SecSource("PoolRASource",
                                              eventEmbeddingSourceParameters,
                                              fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/hydjet_sim_x2_c10_d20080425/hydjet_sim_x2_c10_d20080425_r000002.root')
                                              )
                        )

