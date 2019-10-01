import FWCore.ParameterSet.Config as cms

pixelClusterTagInfos = cms.EDProducer("PixelClusterTagInfoProducer",
    jets     = cms.InputTag("ak4PFJetsCHS"),
    vertices = cms.InputTag("offlinePrimaryVertices"),
    pixelhit = cms.InputTag("siPixelClusters"),
    isPhase1 = cms.bool(True),
    addForward = cms.bool(True),
    minAdcCount = cms.int32(-1), # set to -1 to remove cut
    minJetPtCut = cms.double(100.),
    maxJetEtaCut = cms.double(2.5),
    hadronMass = cms.double(12.), # indicative mass to derive the shrinking cone radius
)

# Use 3 layers instead of 4 in Phase-0 pixel
from Configuration.Eras.Modifier_pixel_2016_cff import pixel_2016
pixel_2016.toModify(pixelClusterTagInfos, isPhase1 = False )

## FastSim modifier (do not run pixelClusterTagInfos because the siPixelCluster input collection is not produced in FastSim)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(pixelClusterTagInfos, minJetPtCut = 1.e+9 )

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(pixelClusterTagInfos, minJetPtCut = 1.e+9 )
    
from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
run2_miniAOD_94XFall17.toModify(pixelClusterTagInfos, minJetPtCut = 1.e+9 )

