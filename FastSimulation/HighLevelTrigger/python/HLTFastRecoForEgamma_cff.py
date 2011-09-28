import FWCore.ParameterSet.Config as cms

# Seeding for electrons (activities)
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronActivitySequenceForHLT_cff import *
# Seeding for electrons (small windows)
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1IsoSequenceForHLT_cff import *
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1NonIsoSequenceForHLT_cff import *
from FastSimulation.EgammaElectronAlgos.pixelMatch3HitElectronSequencesForHLT_cff import *
# Seeding for electrons (large windows)
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1IsoLargeWindowSequenceForHLT_cff import *
from FastSimulation.EgammaElectronAlgos.pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT_cff import *

# Tracking for electrons (activities)
from FastSimulation.EgammaElectronAlgos.ecalActivityEgammaRegionalRecoTracker_cff import *
# Tracking for electrons (small windows)
from FastSimulation.EgammaElectronAlgos.l1IsoElectronsRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1NonIsoElectronsRegionalRecoTracker_cff import *
#from FastSimulation.EgammaElectronAlgos.electrons3HitRegionalRecoTracker_cff import *
# Tracking for electrons (large windows)
from FastSimulation.EgammaElectronAlgos.l1IsoLargeWindowElectronsRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1NonIsoLargeWindowElectronsRegionalRecoTracker_cff import *
# Tracking for photons
from FastSimulation.EgammaElectronAlgos.l1IsoEgammaRegionalRecoTracker_cff import *
from FastSimulation.EgammaElectronAlgos.l1NonIsoEgammaRegionalRecoTracker_cff import *


