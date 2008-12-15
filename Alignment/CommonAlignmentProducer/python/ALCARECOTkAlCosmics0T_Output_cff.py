# last update: $Date: 2008/07/25 11:56:56 $ by $Author: emiglior $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmics0T = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0T', 
            'pathALCARECOTkAlCosmicsCosmicTF0T', 
            'pathALCARECOTkAlCosmicsRS0T')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmics*0T_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', # for cosmics keep also L1
        'keep Si*Cluster*_si*Clusters_*_*', # for cosmics keep original clusters
        'keep *_MEtoEDMConverter_*_*')
)

