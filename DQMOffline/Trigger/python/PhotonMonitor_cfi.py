import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.photonMonitoring_cfi import photonMonitoring

hltPhotonmonitoring = photonMonitoring.clone(
    FolderName = 'HLT/Photon/Photon200/',
    met       = "pfMet", # pfMet
    jets      = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
    electrons = "gedGsfElectrons", # while pfIsolatedElectronsEI are reco::PFCandidate !
    photons = "gedPhotons", # while pfIsolatedElectronsEI are reco::PFCandidate !

    histoPSet = dict(
        lsPSet = dict(
            nbins =  250,
            xmin  =   0.,
            xmax  =  2500.),

        photonPSet = dict(
            nbins =  500 ,
            xmin  =  0.0,
            xmax  = 5000)
    ),

    numGenericTriggerEventPSet = dict(
        andOr         = False,
        #dbLabel       = "ExoDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
        andOrHlt      = True, # True:=OR; False:=AND
        hltInputTag   = "TriggerResults::HLT",
        hltPaths      = ["HLT_Photon175_v*"], # HLT_ZeroBias_v*
        #hltDBKey      = "EXO_HLT_MET",
        errorReplyHlt =  False,
        verbosityLevel = 1),


    denGenericTriggerEventPSet = dict(
        andOr         =  False,
        andOrHlt      = True,
        hltInputTag   = "TriggerResults::HLT",
        hltPaths      = ["HLT_PFJet40_v*","HLT_PFJet60_v*","HLT_PFJet80_v*"], # HLT_ZeroBias_v*
        errorReplyHlt =  False,
        dcsInputTag   = "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions = [ 24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
        andOrDcs      =  False,
        errorReplyDcs = True,
        verbosityLevel = 1)
)

