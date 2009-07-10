# last update: $Date: 2009/01/31 06:40:24 $ by $Author: emiglior $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmics0T_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0T', 
            'pathALCARECOTkAlCosmicsCosmicTF0T', 
            'pathALCARECOTkAlCosmicsRS0T')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlCosmics*0T_*_*', 
        'keep *_eventAuxiliaryHistoryProducer_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', # for cosmics keep also L1
        'keep L1MuGMTReadoutCollection_gtDigis_*_*', 
        'keep Si*Cluster*_si*Clusters_*_*', # for cosmics keep original clusters
        'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOTkAlCosmics0T = copy.deepcopy(OutALCARECOTkAlCosmics0T_noDrop)
OutALCARECOTkAlCosmics0T.outputCommands.insert(0, "drop *")
