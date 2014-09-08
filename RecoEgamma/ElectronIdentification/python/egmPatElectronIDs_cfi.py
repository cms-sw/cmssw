import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

from PhysicsTools.SelectorUtils.trivialCutFlow_cff import *

trivialCutFlowMD5 = central_id_registry.getMD5FromName(trivialCutFlow.idName)

egmPatElectronIDs = cms.EDProducer(
    "VersionedPatElectronIdProducer",
    physicsObjectSrc = cms.InputTag('patElectrons'),
    physicsObjectIDs = cms.VPSet( cms.PSet( idDefinition = trivialCutFlow,
                                            idMD5 = cms.string(trivialCutFlowMD5) )
                           )
)
    
