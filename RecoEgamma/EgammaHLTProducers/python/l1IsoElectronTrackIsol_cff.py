import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltElectronTrackIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltElectronTrackIsol_cfi import *
l1IsoElectronTrackIsol = copy.deepcopy(hltElectronTrackIsol)
l1IsoElectronTrackIsol.electronProducer = 'pixelMatchElectronsL1IsoForHLT'
l1IsoElectronTrackIsol.trackProducer = 'l1IsoElectronsRegionalCTFFinalFitWithMaterial'

