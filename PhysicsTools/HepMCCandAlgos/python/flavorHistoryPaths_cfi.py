import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.flavorHistoryProducer_cfi import *
from PhysicsTools.HepMCCandAlgos.flavorHistoryFilter_cfi import *

# Set up correct sequence for flavorHistoryFilter
flavorHistorySeq = cms.Sequence(bFlavorHistoryProducer*cFlavorHistoryProducer*flavorHistoryFilter)
