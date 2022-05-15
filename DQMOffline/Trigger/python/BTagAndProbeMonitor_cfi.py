import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BTagAndProbeMonitoring_cfi import BTagAndProbeMonitoring

BTagAndProbeMonitoring = BTagAndProbeMonitoring.clone(
    FolderName = 'HLT/BTV/default/',
    requireValidHLTPaths = True,
    applyLeptonPVcuts = False,
    leptonPVcuts = dict(dxy = 9999., 
                        dz  = 9999.),
    
    electrons = "gedGsfElectrons", # while pfIsolatedElectronsEI are reco::PFCandidate !
    elecID    = "egmGsfElectronIDsForDQM:cutBasedElectronID-Fall17-94X-V1-tight", #Electron ID
    muons     = "muons", # while pfIsolatedMuonsEI are reco::PFCandidate !
    vertices  = "offlinePrimaryVertices",
    
    btagAlgos = ['pfDeepCSVJetTags:probb', 'pfDeepCSVJetTags:probbb'],
    workingpoint = 0.8484, # Medium wp
    leptJetDeltaRmin = 0.4,
    bJetDeltaEtaMax  = 9999.,
    
    #genericTriggerEventPSet = dict(andOr = False,
    #                               andOrHlt = True,
    #                               hltInputTag = "TriggerResults::HLT",
    #                               errorReplyHlt = False,
    #                               verbosityLevel = 0,
    #                              ),
   
    genericTriggerEventPSet = dict( andOr         = False,
                                       andOrHlt      = True, # True:=OR; False:=AND
                                       hltInputTag   = "TriggerResults::HLT",
                                       errorReplyHlt = False,
                                       dcsInputTag   = "scalersRawToDigi",
                                       dcsPartitions = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                       andOrDcs      = False,
                                       errorReplyDcs = True,
                                       verbosityLevel = 0,),
)

