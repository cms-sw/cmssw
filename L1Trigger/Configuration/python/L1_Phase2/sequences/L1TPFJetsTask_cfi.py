import FWCore.ParameterSet.Config as cms

from ..modules.l1tLayer2Deregionizer_cfi import *
from ..modules.l1tSC4PFL1PF_cfi import *
from ..modules.l1tSC4PFL1Puppi_cfi import *
from ..modules.l1tSC4PFL1PuppiCorrected_cfi import *
from ..modules.l1tSC4PFL1PuppiCorrectedEmulator_cfi import *
from ..modules.l1tSC4PFL1PuppiCorrectedEmulatorMHT_cfi import *
from ..modules.l1tSC4PFL1PuppiEmulator_cfi import *
from ..modules.l1tSC8PFL1Puppi_cfi import *
from ..modules.l1tSC8PFL1PuppiCorrected_cfi import *
from ..modules.l1tSC8PFL1PuppiCorrectedEmulator_cfi import *
from ..modules.l1tSC8PFL1PuppiEmulator_cfi import *

L1TPFJetsTask = cms.Task(l1tLayer2Deregionizer, l1tSC4PFL1PF, l1tSC4PFL1Puppi, l1tSC4PFL1PuppiCorrected, l1tSC4PFL1PuppiCorrectedEmulator, l1tSC4PFL1PuppiCorrectedEmulatorMHT, l1tSC4PFL1PuppiEmulator, l1tSC8PFL1Puppi, l1tSC8PFL1PuppiCorrected, l1tSC8PFL1PuppiCorrectedEmulator, l1tSC8PFL1PuppiEmulator)
