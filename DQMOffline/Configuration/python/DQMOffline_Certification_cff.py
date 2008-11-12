import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMDaqInfo_cfi import *

DQMOffline_Certification = cms.Sequence(dqmDaqInfo)


