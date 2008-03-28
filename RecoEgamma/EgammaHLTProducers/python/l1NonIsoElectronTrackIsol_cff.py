import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltElectronTrackIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltElectronTrackIsol_cfi import *
l1NonIsoElectronTrackIsol = copy.deepcopy(hltElectronTrackIsol)
l1NonIsoElectronTrackIsol.electronProducer = 'pixelMatchElectronsL1NonIsoForHLT'
l1NonIsoElectronTrackIsol.trackProducer = 'l1NonIsoElectronsRegionalCTFFinalFitWithMaterial'

