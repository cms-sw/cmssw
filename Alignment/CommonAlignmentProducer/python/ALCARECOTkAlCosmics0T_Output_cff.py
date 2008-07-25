# last update: $Date: 2008/06/20 14:13:40 $ by $Author: flucke $

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
        'keep Si*Cluster*_*_*_*', # for cosmics keep also clusters
        'keep *_MEtoEDMConverter_*_*')
)

