import FWCore.ParameterSet.Config as cms

##################
# Electron ID ######
from DPGAnalysis.Skims.WElectronSkim_cff import *


#  GsfElectron ################ 

looseElectronSelection = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag( ELECTRON_COLL ),
    cut = cms.string( ELECTRON_CUTS )    
)


#####################################
#####################################

from HLTrigger.HLTfilters.hltHighLevel_cfi import *

hltBtagTopMuEGSelection = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring(
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepJet_1p5_v*',  # DeepCSV paths not available anymore. See https://its.cern.ch/jira/browse/CMSHLT-2592 
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PNet2BTagMean0p50_v*',  # Taken from HLTrigger/Configuration/python/HLTrigger_Datasets_GRun_cff.py
     ), 
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True), # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False), # throw exception on unknown path names
#     saveTags = cms.bool(False)
)


#####################################
############ MU SELECTION #########################
 
muonSelection = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("muons"),
                             cut = cms.string('pt > 20 && abs(eta)<2.4 && isGlobalMuon = 1 && isTrackerMuon = 1 && abs(innerTrack().dxy)<2.0'),
                             filter = cms.bool(True)
)



muonDecayProducer = cms.EDProducer("CandViewShallowCloneCombiner",
                         checkCharge = cms.bool(False),
                         cut = cms.string('mass > 0'),
                         decay = cms.string("muonSelection looseElectronSelection")
                         )
muonDecaySelection = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("muonDecayProducer"),
                             minNumber = cms.uint32(1)
                             )
                             

############################################
################# DI JET FILTER ###########################

AK4CandidateJetProducer = cms.EDFilter("EtaPtMinCandViewSelector",
                      src = cms.InputTag("ak4PFJets"),
                      ptMin = cms.double(30),
                      etaMin = cms.double(-2.5),
                      etaMax = cms.double(2.5)
                      )
			                                
dijetSelection = cms.EDFilter("CandViewCountFilter",
			   src = cms.InputTag("AK4CandidateJetProducer"),
			   minNumber = cms.uint32(2)
			   )


TopMuEGsequence = cms.Sequence(hltBtagTopMuEGSelection * muonSelection * looseElectronSelection * muonDecayProducer * muonDecaySelection * AK4CandidateJetProducer * dijetSelection)
