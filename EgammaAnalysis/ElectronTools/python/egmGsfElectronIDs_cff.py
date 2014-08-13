from EgammaAnalysis.ElectronTools.egmGsfElectronIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

#simple ID
from PhysicsTools.SelectorUtils.trivialCutFlow_cff import trivialCutFlow
trivialCutFlowMD5 = central_id_registry.getMD5FromName(trivialCutFlow.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = trivialCutFlow,
              idMD5 = cms.string(trivialCutFlowMD5) )
    )

#CSA14 tight ID for 50ns
from EgammaAnalysis.ElectronTools.cutBasedElectronID_CSA14_50ns_V0_cff import cutBasedElectronID_CSA14_50ns_V0_standalone_tight
csa14_50ns_tight_md5 = central_id_registry.getMD5FromName(cutBasedElectronID_CSA14_50ns_V0_standalone_tight.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = cutBasedElectronID_CSA14_50ns_V0_standalone_tight,
              idMD5 = cms.string(csa14_50ns_tight_md5) )
    )
    
egmGsfElectronIDSequence = cms.Sequence(egmGsfElectronIDs)
