import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.htMonitoring_cfi import htMonitoring

# config file for monitoring the trigger efficiency vs invariant dijetmass of the two leading jets
# see python/HTMonitor_cfi.py or plugins/HTMonitor.h or plugins/HTMonitor.cc for more details

hltMjjmonitoring = htMonitoring.clone(
    FolderName = 'HLT/HT/PFMETNoMu120/',
    quantity = 'Mjj', # set quantity to invariant dijetmass
    jetSelection = "pt > 200 && eta < 2.4",
    dEtaCut     = 1.3,
    met       = "pfMet",
    jets      = "ak8PFJetsPuppi",
    electrons = "gedGsfElectrons",
    muons     = "muons",
    
    histoPSet = dict(htPSet = dict(
            nbins =  200,
            xmin  =  -0.5,
            xmax  = 19999.5)),

    numGenericTriggerEventPSet = dict(
        andOr         =  False,
        andOrHlt      = True, # True:=OR; False:=AND
        hltInputTag   =  "TriggerResults::HLT",
        hltPaths      = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"],
        errorReplyHlt = False,
        verbosityLevel = 0),

    denGenericTriggerEventPSet = dict(
        andOr         =  False,
        dcsInputTag   =  "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions = [ 24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29
        andOrDcs      =  False,
        errorReplyDcs = True,
        verbosityLevel = 0,
        hltPaths      = ["HLT_IsoMu27_v*"])
)

