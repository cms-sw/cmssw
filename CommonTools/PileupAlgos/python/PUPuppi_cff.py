import FWCore.ParameterSet.Config as cms

from CommonTools.PileupAlgos.Puppi_cff import *

pupuppi             = puppi.clone()
pupuppi.invertPuppi = True

