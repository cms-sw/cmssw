import FWCore.ParameterSet.Config as cms

#produce the mag field
#produce the associator
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByPosition_cff import *
#need a propagator in case analysis is made a radius!=0
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
#produce the module
from RecoMuon.TrackingTools.MuonErrorMatrixAnalyzer_cfi import *


