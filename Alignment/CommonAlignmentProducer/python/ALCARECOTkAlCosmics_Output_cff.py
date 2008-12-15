# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2008/07/25 11:56:56 $ by $Author: emiglior $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmics = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF', 
            'pathALCARECOTkAlCosmicsCosmicTF', 
            'pathALCARECOTkAlCosmicsRS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
#        'keep *_ALCARECOTkAlCosmics*_*_*', # keeps also 0T ones if in same job
        'keep *_ALCARECOTkAlCosmicsCTF_*_*', 
        'keep *_ALCARECOTkAlCosmicsCosmicTF_*_*', 
        'keep *_ALCARECOTkAlCosmicsRS_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', # for cosmics keep also L1
        'keep Si*Cluster*_si*Clusters_*_*', # for cosmics keep original clusters
        'keep *_MEtoEDMConverter_*_*')
)

