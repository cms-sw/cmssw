import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
#pieces needed to run the associator-related stuff
# from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from RecoMuon.MuonIsolationProducers.isoDepositProducerIOBlocks_cff import *
from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.jetExtractorBlock_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorBlocks_cff import *


