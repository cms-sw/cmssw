# Misc loads for VID framework
from EgammaAnalysis.ElectronTools.egmGsfElectronIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever 
# else is needed for IDs
from EgammaAnalysis.ElectronTools.ElectronIDValueMapProducer_cfi import *

#simple ID
from PhysicsTools.SelectorUtils.trivialCutFlow_cff import trivialCutFlow
trivialCutFlowMD5 = central_id_registry.getMD5FromName(trivialCutFlow.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = trivialCutFlow,
              idMD5 = cms.string(trivialCutFlowMD5) )
    )

#
#CSA14 tight ID for 50ns
#

# V0: preliminary safe cuts from Giovanni Zevi Dalla Porta
# Comment it out, superseeded by V1 below 
#
# from EgammaAnalysis.ElectronTools.cutBasedElectronID_CSA14_50ns_V0_cff import cutBasedElectronID_CSA14_50ns_V0_standalone_tight
# csa14_50ns_tight_md5 = central_id_registry.getMD5FromName(cutBasedElectronID_CSA14_50ns_V0_standalone_tight.idName)
# egmGsfElectronIDs.electronIDs.append( 
#     cms.PSet( idDefinition = cutBasedElectronID_CSA14_50ns_V0_standalone_tight,
#               idMD5 = cms.string(csa14_50ns_tight_md5) )
#     )
    
# V1: tuned cuts for this scenario
from EgammaAnalysis.ElectronTools.Identification.cutBasedElectronID_CSA14_50ns_V1_cff \
import cutBasedElectronID_CSA14_50ns_V1_standalone_veto
csa14_50ns_veto_md5_v1 = central_id_registry.getMD5FromName(cutBasedElectronID_CSA14_50ns_V1_standalone_veto.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = cutBasedElectronID_CSA14_50ns_V1_standalone_veto,
              idMD5 = cms.string(csa14_50ns_veto_md5_v1) )
    )
    
from EgammaAnalysis.ElectronTools.Identification.cutBasedElectronID_CSA14_50ns_V1_cff \
import cutBasedElectronID_CSA14_50ns_V1_standalone_loose
csa14_50ns_loose_md5_v1 = central_id_registry.getMD5FromName(cutBasedElectronID_CSA14_50ns_V1_standalone_loose.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = cutBasedElectronID_CSA14_50ns_V1_standalone_loose,
              idMD5 = cms.string(csa14_50ns_loose_md5_v1) )
    )
    
from EgammaAnalysis.ElectronTools.Identification.cutBasedElectronID_CSA14_50ns_V1_cff \
import cutBasedElectronID_CSA14_50ns_V1_standalone_medium
csa14_50ns_medium_md5_v1 = central_id_registry.getMD5FromName(cutBasedElectronID_CSA14_50ns_V1_standalone_medium.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = cutBasedElectronID_CSA14_50ns_V1_standalone_medium,
              idMD5 = cms.string(csa14_50ns_medium_md5_v1) )
    )
    
from EgammaAnalysis.ElectronTools.Identification.cutBasedElectronID_CSA14_50ns_V1_cff \
import cutBasedElectronID_CSA14_50ns_V1_standalone_tight
csa14_50ns_tight_md5_v1 = central_id_registry.getMD5FromName(cutBasedElectronID_CSA14_50ns_V1_standalone_tight.idName)
egmGsfElectronIDs.electronIDs.append( 
    cms.PSet( idDefinition = cutBasedElectronID_CSA14_50ns_V1_standalone_tight,
              idMD5 = cms.string(csa14_50ns_tight_md5_v1) )
    )
    
egmGsfElectronIDSequence = cms.Sequence(electronIDValueMapProducer * egmGsfElectronIDs)
