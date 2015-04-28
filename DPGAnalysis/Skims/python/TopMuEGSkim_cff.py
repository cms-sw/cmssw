import FWCore.ParameterSet.Config as cms

	
HLTPath = "HLT_Ele*"
HLTProcessName = "HLT"

### cut on electron tag
#ELECTRON_ET_CUT_MIN = 10.0
ELECTRON_ET_CUT_MIN_TIGHT = 20.0
ELECTRON_ET_CUT_MIN_LOOSE = 10.0
ELECTRON_COLL = "gedGsfElectrons"
ELECTRON_CUTS = "(abs(superCluster.eta)<2.5) && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN_LOOSE) + ")"

MASS_CUT_MIN = 0.

##################
# Electron ID ######
from DPGAnalysis.Skims.WElectronSkim_cff import *


#  GsfElectron ################ 

ElectronPassingVeryLooseId = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag( ELECTRON_COLL ),
    cut = cms.string( ELECTRON_CUTS )    
)


#####################################
#####################################
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

ZEM_DiJetHltFilter = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*','HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*'), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True), # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False), # throw exception on unknown path names
#     saveTags = cms.bool(False)
 )


#elec_sequence = cms.Sequence(
#    ZEM_DiJetHltFilter * 
#    goodElectrons *
#    PassingVeryLooseId
#    )

#####################################
############ MU SELECTION #########################
 

# Get muons of needed quality for Zs
looseMuonsForTop = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("muons"),
                             cut = cms.string('pt > 20 && abs(eta)<2.4 && isGlobalMuon = 1 && isTrackerMuon = 1 && abs(innerTrack().dxy)<2.0'),
                             filter = cms.bool(True)
)



elecMuon = cms.EDProducer("CandViewShallowCloneCombiner",
                         checkCharge = cms.bool(False),
                         cut = cms.string('mass > 0'),
                         decay = cms.string("looseMuonsForTop ElectronPassingVeryLooseId")
                         )
elecMuonFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("elecMuon"),
                             minNumber = cms.uint32(1)
                             )
                             

############################################
################# DI JET FILTER ###########################

import FWCore.ParameterSet.Config as cms


Jet1 = cms.EDFilter("EtaPtMinCandViewSelector",
                      src = cms.InputTag("ak4PFJets"),
                      ptMin = cms.double(30),
                      etaMin = cms.double(-2.5),
                      etaMax = cms.double(2.5)
                      )
			                                
dijetFilter = cms.EDFilter("CandViewCountFilter",
			   src = cms.InputTag("Jet1"),
			   minNumber = cms.uint32(2)
			   )


TopMuEGsequence = cms.Sequence(ZEM_DiJetHltFilter * looseMuonsForTop * ElectronPassingVeryLooseId * elecMuon * elecMuonFilter * Jet1 * dijetFilter)

#from Configuration.EventContent.EventContent_cff import OutALCARECOEcalCalElectron
#TopMuEGSkimContent = OutALCARECOEcalCalElectron.clone()
#TopMuEGSkimContent.outputCommands.extend( [ 
#  "keep *drop *",
#  "keep *_pfMet_*_*", 
#  "keep *_MuEG_Skim_*"
# ] )
