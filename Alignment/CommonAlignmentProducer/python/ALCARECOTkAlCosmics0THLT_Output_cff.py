# last update: $Date: 2008/05/09 15:18:17 $ by $Author: emiglior $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmics0THLT = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0THLT', 
            'pathALCARECOTkAlCosmicsCosmicTF0THLT', 
            'pathALCARECOTkAlCosmicsRS0THLT')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmics*0T_*_*', 
        'keep Si*Cluster*_*_*_*', # for cosmics keep also clusters
        'keep *_MEtoEDMConverter_*_*')
)

