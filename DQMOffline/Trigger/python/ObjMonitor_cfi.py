import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.objMonitoring_cfi import objMonitoring

hltobjmonitoring = objMonitoring.clone(
    FolderName = 'HLT/GENERIC/',
    doMETHistos = True,
    met       = "pfMet",
    jets      = "ak4PFJetsCHS",
    electrons = "gedGsfElectrons",
    muons     = "muons",
    photons   = "gedPhotons",
    tracks    = "generalTracks",
    doJetHistos = True,
    doHTHistos = True,
    doHMesonGammaHistos = False,

    histoPSet = dict(
            metPSet = dict(
                    nbins =  200  ,
                    xmin  =   -0.5,
                    xmax  = 19999.5),

            phiPSet = dict(
                    nbins =  64  ,
                    xmin  =   -3.1416,
                    xmax  = 3.1416),

            jetetaPSet = dict(
                    nbins = 100 ,
                    xmin  = -5,
                    xmax  = 5),

            detajjPSet = dict(
                    nbins = 90 ,
                    xmin  = 0,
                    xmax  = 9),

            dphijjPSet = dict(
                    nbins =  64 ,
                    xmin  =  0,
                    xmax  = 3.1416),

            mindphijmetPSet = dict(
                    nbins =  64,
                    xmin  =  0,
                    xmax  = 3.1416),

            htPSet = dict(
                    nbins = 60  ,
                    xmin  = -0.5,
                    xmax  = 1499.5),

            hmgetaPSet = dict(
                    nbins = 60  ,
                    xmin  = -2.6,
                    xmax  = 2.6),
        ),

    numGenericTriggerEventPSet = dict(
        andOr         = False,
        #dbLabel       = "ExoDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
        andOrHlt      = True, # True:=OR; False:=AND
        hltInputTag   =  "TriggerResults::HLT",
        hltPaths      = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"], # HLT_ZeroBias_v*
        #hltDBKey      = "EXO_HLT_MET",
        errorReplyHlt =  False,
        verbosityLevel = 1),

    denGenericTriggerEventPSet = dict(
        andOr         =  False,
        dcsInputTag   =  "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions = [ 24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
        andOrDcs      =  False,
        errorReplyDcs = True,
        verbosityLevel = 1)
)


