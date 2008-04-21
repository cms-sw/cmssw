import FWCore.ParameterSet.Config as cms

from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1IsoSequenceForHLT_cff import *
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1NonIsoSequenceForHLT_cff import *
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1IsoLargeWindowSequenceForHLT_cff import *
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT_cff import *
from FastSimulation.EgammaElectronAlgos.l1IsoElectronsRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1NonIsoElectronsRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1IsoLargeWindowElectronsRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1NonIsoLargeWindowElectronsRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1IsoEgammaRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1NonIsoEgammaRegionalRecoTracker_cff import *
l1seedSingle.L1MuonCollectionTag = 'l1ParamMuons'
l1seedRelaxedSingle.L1MuonCollectionTag = 'l1ParamMuons'
l1seedDouble.L1MuonCollectionTag = 'l1ParamMuons'
l1seedRelaxedDouble.L1MuonCollectionTag = 'l1ParamMuons'
l1seedExclusiveDouble.L1MuonCollectionTag = 'l1ParamMuons'
l1seedSinglePrescaled.L1MuonCollectionTag = 'l1ParamMuons'
l1seedSingle.L1GtObjectMapTag = 'gtDigis'
l1seedRelaxedSingle.L1GtObjectMapTag = 'gtDigis'
l1seedDouble.L1GtObjectMapTag = 'gtDigis'
l1seedRelaxedDouble.L1GtObjectMapTag = 'gtDigis'
l1seedExclusiveDouble.L1GtObjectMapTag = 'gtDigis'
l1seedSinglePrescaled.L1GtObjectMapTag = 'gtDigis'

