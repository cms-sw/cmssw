import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.htMonitoring_cfi import htMonitoring

# config file for monitoring the trigger efficiency vs softdropmass of the leading jet
# see python/HTMonitor_cfi.py or plugins/HTMonitor.h or plugins/HTMonitor.cc for more details

hltSoftdropmonitoring = htMonitoring.clone(
    FolderName = 'HLT/HT/PFMETNoMu120/',
    quantity = 'softdrop', # set quantity to leading jet softdropmass
    jetSelection = "pt > 65 && eta < 2.4",
    dEtaCut     = 1.3,
    met       = "pfMet",
    jets      = "ak8PFJetsPuppiSoftDrop", # dont set this to non-SoftdropJets
    electrons = "gedGsfElectrons",
    muons     = "muons",
    histoPSet = dict(htBinning = [0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90., 95., 100., 105., 110., 115., 120., 125., 130., 135., 140., 145., 150., 155., 160., 165., 170., 175., 180., 185., 190., 195., 200.],
                     htPSet = dict(
                            nbins =  200,
                            xmin  = -0.5,
                            xmax  = 19999.5)),
    
numGenericTriggerEventPSet = dict(
    andOr  = False,
    andOrHlt      = True, # True:=OR; False:=AND
    hltInputTag   = "TriggerResults::HLT",
    hltPaths      = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"],
    errorReplyHlt = False,
    verbosityLevel = 0),

denGenericTriggerEventPSet = dict(
    andOr  = False,
    dcsInputTag   = "scalersRawToDigi",
    dcsRecordInputTag = "onlineMetaDataDigis",
    dcsPartitions = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29
    andOrDcs      = False,
    errorReplyDcs = True,
    verbosityLevel = 0,
    hltPaths      = ["HLT_IsoMu27_v*"])
)

