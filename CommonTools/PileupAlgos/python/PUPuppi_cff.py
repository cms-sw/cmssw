import FWCore.ParameterSet.Config as cms

from CommonTools.PileupAlgos.Puppi_cff import *

pupuppi = puppi.clone(
    invertPuppi = True
)

