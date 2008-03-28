import FWCore.ParameterSet.Config as cms

# ECAL Super clusters
from RecoEgamma.EgammaHLTProducers.l1IsolatedEcalSuperClusters_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsolatedEcalSuperClusters_cff import *
# ECAL Rec Hits
from RecoEgamma.EgammaHLTProducers.l1IsoRecoEcalCandidate_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoRecoEcalCandidate_cff import *
# HCAL Rec Hits
from RecoEgamma.EgammaHLTProducers.l1IsolatedElectronHcalIsol_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsolatedElectronHcalIsol_cff import *
# Tracking sequences
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronL1IsoTrackingSequenceForHLT_cff import *
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronL1NonIsoTrackingSequenceForHLT_cff import *
# Electron Track isolation
from RecoEgamma.EgammaHLTProducers.l1IsoElectronTrackIsol_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoElectronTrackIsol_cff import *
# Electron ECAL/HCAL isolation
# include "RecoEgamma/EgammaHLTProducers/data/hltHcalDoubleCone.cfi" 
from RecoEgamma.EgammaHLTProducers.l1NonIsoEMHcalDoubleCone_cff import *
# Photon ECAL isolation
from RecoEgamma.EgammaHLTProducers.l1IsoPhotonEcalIsol_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoPhotonEcalIsol_cff import *
# Photon HCAL isolation
from RecoEgamma.EgammaHLTProducers.l1IsolatedPhotonHcalIsol_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsolatedPhotonHcalIsol_cff import *
# Photon Track isolation
from RecoEgamma.EgammaHLTProducers.l1IsoPhotonTrackIsol_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoPhotonTrackIsol_cff import *
# Electron track isolation for large seeding windows
from RecoEgamma.EgammaHLTProducers.l1IsoLargeWindowElectronTrackIsol_cff import *
from RecoEgamma.EgammaHLTProducers.l1NonIsoLargeWindowElectronTrackIsol_cff import *

