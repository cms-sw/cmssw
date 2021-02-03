import FWCore.ParameterSet.Config as cms

from ..modules.clusterSummaryProducer_cfi import *
from ..tasks.pixeltrackerlocalrecoTask_cfi import *

trackerlocalrecoTask = cms.Task(clusterSummaryProducer, pixeltrackerlocalrecoTask)
