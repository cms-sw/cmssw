import FWCore.ParameterSet.Config as cms

from ..modules.muonSeededTracksOutInDisplacedClassifier_cfi import *
from ..tasks.muonSeededStepExtraInOutTask_cfi import *

muonSeededStepExtraDisplacedTask = cms.Task(muonSeededStepExtraInOutTask, muonSeededTracksOutInDisplacedClassifier)
