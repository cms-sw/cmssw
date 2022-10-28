import FWCore.ParameterSet.Config as cms

from DQM.GEM.gemEffByGEMCSCSegmentSource_cfi import *
from DQM.GEM.gemEffByGEMCSCSegmentClient_cfi import *

gemEffByGEMCSCSegment = cms.Sequence(
    gemEffByGEMCSCSegmentSource *
    gemEffByGEMCSCSegmentClient
)
