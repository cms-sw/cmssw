import FWCore.ParameterSet.Config as cms

from ..tasks.pixeltrackerlocalrecoTask_cfi import *

trackerlocalrecoTask = cms.Task(
    pixeltrackerlocalrecoTask
)
