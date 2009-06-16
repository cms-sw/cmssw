import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.flavorHistoryProducer_cfi import *
from PhysicsTools.HepMCCandAlgos.flavorHistoryFilter_cfi import *


from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.sisCone5GenJets_cff import *

# Set up correct sequence for flavorHistoryFilter
flavorHistorySeq = cms.Sequence(genJetParticles*sisCone5GenJets*
                                bFlavorHistoryProducer*cFlavorHistoryProducer*flavorHistoryFilter)
