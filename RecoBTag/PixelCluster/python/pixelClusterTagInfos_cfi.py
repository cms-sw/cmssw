import FWCore.ParameterSet.Config as cms

# This computer counts the number of PROMPT tracks in a jet.
# i.e. The tagging variable it calculates is equal to this number.
# Its main use it for exotica physics, not b tagging.
# It counts tracks with impact parameter significance less than some cut.
# If you also wish to apply a cut on the maximum allowed impact parameter,
# you can do this in the TagInfoProducer.

pixelClusterTagInfos = cms.ESProducer("PixelClusterTagInfoProducer",
    jetsAK4 = cms.InputTag("ak4PFJetsCHS"),
    jetsAK8 = cms.InputTag("ak8PFJetsCHS"),
    vertices = cms.InputTag("offlinePrimaryVertices"),
    pixelhit = cms.InputTag("siPixelClusters"),
    minJetPtCut = cms.double(100.),
    minAdcCount = cms.uint32(0), # set to 0 to remove cut
)

