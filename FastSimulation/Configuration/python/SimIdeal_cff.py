import FWCore.ParameterSet.Config as cms

# need for any other modules from commoninputs?

# FastSim SimHits producer
from FastSimulation.SimplifiedGeometryPropagator.fastSimProducer_cff import *

# Gaussian Smearing RecHit producer
from FastSimulation.TrackingRecHitProducer.TrackingRecHitProducer_cfi import *

# Muon simHit sequence
from FastSimulation.MuonSimHitProducer.MuonSimHitProducer_cfi import *

# propagors from muon reconstruction are used in muon simulation
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

psim = cms.Sequence(
    fastSimProducer+
    MuonSimHits
    )
