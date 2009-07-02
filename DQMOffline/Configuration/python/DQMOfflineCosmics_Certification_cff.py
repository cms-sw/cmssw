import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMDaqInfo_cfi import *

DQMOfflineCosmics_Certification = cms.Sequence(dqmDaqInfo)


