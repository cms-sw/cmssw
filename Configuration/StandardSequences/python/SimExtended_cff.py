import FWCore.ParameterSet.Config as cms

# Hector transport of particles along the beam pipe for very forward detectors
from SimTransport.HectorProducer.HectorTransport_cfi import *

# CMSSW/Geant4 interface
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Generator.HepMCProductLabel = cms.string('LHCTransport')
g4SimHits.StackingAction.MaxTimeNames = cms.vstring('ZDCRegion')
g4SimHits.StackingAction.MaxTrackTimes = cms.vdouble(2000.0)
g4SimHits.SteppingAction.MaxTimeNames = cms.vstring('ZDCRegion')
g4SimHits.SteppingAction.MaxTrackTimes = cms.vdouble(2000.0)
g4SimHits.CaloSD.HCNames = cms.vstring('EcalHitsEB','EcalHitsEE','EcalHitsES','HcalHits','ZDCHITS')
g4SimHits.CaloSD.EminHits = cms.vdouble(0.015,0.010,0.0,0.0,0.0)
g4SimHits.CaloSD.TmaxHits = cms.vdouble(500.0,500.0,500.0,500.0,2000.0)

psim = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*LHCTransport*g4SimHits)


