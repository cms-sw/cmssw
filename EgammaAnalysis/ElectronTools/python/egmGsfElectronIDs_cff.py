from EgammaAnalysis.ElectronTools.egmGsfElectronIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

#simple ID
from PhysicsTools.SelectorUtils.trivialCutFlow_cff import trivialCutFlow
trivialCutFlowMD5 = central_id_registry.getMD5FromName(trivialCutFlow.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = trivialCutFlow,
              idMD5 = cms.string(trivialCutFlowMD5) )
    )
    
egmGsfElectronIDSequence = cms.Sequence(egmGsfElectronIDs)
