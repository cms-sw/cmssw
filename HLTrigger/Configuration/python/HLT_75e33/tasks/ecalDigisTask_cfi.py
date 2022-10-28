import FWCore.ParameterSet.Config as cms

from ..modules.ecalDigis_cfi import *

ecalDigisTask = cms.Task(
    ecalDigis
)
