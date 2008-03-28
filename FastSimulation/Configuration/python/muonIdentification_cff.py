import FWCore.ParameterSet.Config as cms

# add TrackDetectorAssociator lookup maps to the EventSetup
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from RecoMuon.MuonIdentification.isolation_cff import *
from RecoMuon.MuonIdentification.muons_cfi import *
from RecoMuon.MuonIdentification.calomuons_cfi import *
muonIdProducerSequence = cms.Sequence(muons*calomuons)

