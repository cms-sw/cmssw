# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2008/06/12 20:35:34 $ by $Author: flucke $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmics = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF', 
            'pathALCARECOTkAlCosmicsCosmicTF', 
            'pathALCARECOTkAlCosmicsRS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmics*_*_*', 
        'keep Si*Cluster*_*_*_*', # for cosmics keep also clusters
        'keep *_MEtoEDMConverter_*_*')
)

