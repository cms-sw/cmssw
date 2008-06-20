# last update: $Date: 2008/05/09 15:19:27 $ by $Author: emiglior $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmicsHLT = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTFHLT', 
            'pathALCARECOTkAlCosmicsCosmicTFHLT', 
            'pathALCARECOTkAlCosmicsRSHLT')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmics*_*_*', 
        'keep Si*Cluster*_*_*_*', # for cosmics keep also clusters
        'keep *_MEtoEDMConverter_*_*')
)

