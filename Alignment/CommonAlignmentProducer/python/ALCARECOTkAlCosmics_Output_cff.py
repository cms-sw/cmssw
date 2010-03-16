# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2010/03/12 21:18:57 $ by $Author: emiglior $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmics_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF', 
            'pathALCARECOTkAlCosmicsCosmicTF')
    ),
    outputCommands = cms.untracked.vstring(
#        'keep *_ALCARECOTkAlCosmics*_*_*', # keeps also 0T ones if in same job
        'keep *_ALCARECOTkAlCosmicsCTF_*_*', 
        'keep *_ALCARECOTkAlCosmicsCosmicTF_*_*', 
        'keep siStripDigis_DetIdCollection_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep Si*Cluster*_si*Clusters_*_*', # for cosmics keep original clusters
        'keep recoMuons_muons1Leg_*_*', # save muons as timing info is needed for BP corrections in deconvolution
        'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOTkAlCosmics = copy.deepcopy(OutALCARECOTkAlCosmics_noDrop)
OutALCARECOTkAlCosmics.outputCommands.insert(0, "drop *")
