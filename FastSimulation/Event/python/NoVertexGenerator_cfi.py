import FWCore.ParameterSet.Config as cms

NoVertexGeneratorBlock = cms.PSet(
    VertexGenerator = cms.PSet(
        type = cms.string('None')
    )
)

