# Author     : Andreas Mussgiller
# Date       : July 1st, 2010
# last update: $Date: 2010/07/06 11:48:22 $ by $Author: mussgill $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmicsInCollisions_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsInCollisions')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlCosmicsInCollisions_*_*', 
        'keep siStripDigis_DetIdCollection_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep Si*Cluster*_si*Clusters_*_*', # for cosmics keep original clusters
        'keep recoMuons_muons1Leg_*_*') # save muons as timing info is needed for BP corrections in deconvolution
)

import copy
OutALCARECOTkAlCosmicsInCollisions = copy.deepcopy(OutALCARECOTkAlCosmicsInCollisions_noDrop)
OutALCARECOTkAlCosmicsInCollisions.outputCommands.insert(0, "drop *")
