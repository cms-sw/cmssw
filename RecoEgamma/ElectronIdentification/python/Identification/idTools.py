import FWCore.ParameterSet.Config as cms

def convertIDToRunOnMiniAOD(idPSet):
    idPSet.idName=idPSet.idName.value()+"-miniAOD"
    for pset in idPSet.cutFlow:
        if hasattr(pset,'vertexSrc'):
           pset.vertexSrc = cms.InputTag("offlineSlimmedPrimaryVertices")

