import FWCore.ParameterSet.Config as cms

from ..modules.l1tLayer2DeregionizerExtended_cfi import *
from ..modules.l1tSC4NGJetProducer_cfi import *
from ..modules.l1tSC4PFL1PuppiExtended_cfi import *
from ..modules.l1tSC4PFL1PuppiExtendedCorrectedEmulator_cfi import *
from ..modules.l1tSC4PFL1PuppiExtendedEmulator_cfi import *

L1TPFJetsExtendedTask = cms.Task(l1tLayer2DeregionizerExtended, l1tSC4NGJetProducer, l1tSC4PFL1PuppiExtended, l1tSC4PFL1PuppiExtendedCorrectedEmulator, l1tSC4PFL1PuppiExtendedEmulator)
