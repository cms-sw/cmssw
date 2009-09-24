import FWCore.ParameterSet.Config as cms

# add TrackDetectorAssociator lookup maps to the EventSetup
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from RecoMuon.MuonIdentification.isolation_cff import *
from RecoMuon.MuonIdentification.muons_cfi import *
from RecoMuon.MuonIdentification.calomuons_cfi import *

from RecoMuon.GlobalTrackingTools.GlobalTrackQuality_cfi import *
muons.fillGlobalTrackQuality = True

muonIdProducerSequence = cms.Sequence(glbTrackQual*muons*calomuons)

