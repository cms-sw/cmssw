import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.ElectronIDProducers.electronId_cfi import *
import copy
from EgammaAnalysis.ElectronIDProducers.electronId_cfi import *
electronIdRobust = copy.deepcopy(electronId)
patElectronId = cms.Sequence(electronId*electronIdRobust)
electronIdRobust.doPtdrId = False
electronIdRobust.doCutBased = True
electronId.electronProducer = 'allLayer0Electrons'
electronIdRobust.electronProducer = 'allLayer0Electrons'

