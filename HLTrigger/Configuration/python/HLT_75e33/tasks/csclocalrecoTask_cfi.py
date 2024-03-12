import FWCore.ParameterSet.Config as cms

from ..modules.hltCsc2DRecHits_cfi import *
from ..modules.hltCscSegments_cfi import *

csclocalrecoTask = cms.Task(
    hltCsc2DRecHits,
    hltCscSegments
)
# foo bar baz
# A92tzKT16jSHf
# w2Xs3HufLququ
