import FWCore.ParameterSet.Config as cms

# Seeding and track reconstruction for electrons (small windows)
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronL1IsoSequenceForHLT_cff import *
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronL1NonIsoSequenceForHLT_cff import *
# Seeding and track reconstruction for electrons (large windows)
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronL1IsoLargeWindowSequenceForHLT_cff import *
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT_cff import *
# Tracking for electrons, for track isolation (small windows)
from RecoEgamma.EgammaHLTProducers.l1IsoElectronsRegionalRecoTracker_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoElectronsRegionalRecoTracker_cff import *
# Tracking for electrons, for track isolation (large windows)
from RecoEgamma.EgammaHLTProducers.l1IsoLargeWindowElectronsRegionalRecoTracker_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoLargeWindowElectronsRegionalRecoTracker_cff import *
# Tracking for photons, for track isolation
from RecoEgamma.EgammaHLTProducers.l1IsoEgammaRegionalRecoTracker_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoEgammaRegionalRecoTracker_cff import *

