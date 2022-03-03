import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.dijetMonitoring_cfi import dijetMonitoring
DiPFjetAve40_Prommonitoring = dijetMonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve40/',
    met       = "pfMet", # pfMet
    #pfjets    = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
    dijetSrc  = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
    electrons = "gedGsfElectrons", # while pfIsolatedElectronsEI are reco::PFCandidate !
    muons     = "muons", # while pfIsolatedMuonsEI are reco::PFCandidate !
    ptcut     = 20, # while pfIsolatedMuonsEI are reco::PFCandidate !

    histoPSet = dict(dijetPSet = dict(
            nbins = 200 ,
            xmin  =  0,
            xmax  = 1000.),
                     dijetPtThrPSet = dict(
                             nbins = 50 ,
                             xmin  =  0.,
                             xmax  = 100.)),

    numGenericTriggerEventPSet = dict(andOr = False,
                                      dbLabel = "JetMETDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !                                                                                                                                                        
                                      andOrHlt      = True, # True:=OR; False:=AND                                                                                           
                                      hltInputTag   = "TriggerResults::HLT" ,
                                      hltPaths      = ["HLT_DiPFJetAve40_v*"], # HLT_ZeroBias_v*                                                                             
                                      errorReplyHlt =  False,
                                      verbosityLevel = 1),


    denGenericTriggerEventPSet = dict(andOr = False,
                                      dcsInputTag   = "scalersRawToDigi",
                                      dcsRecordInputTag = "onlineMetaDataDigis",
                                      dcsPartitions = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !                            
                                      andOrDcs      = False,
                                      errorReplyDcs = True,
                                      verbosityLevel = 1)
)
