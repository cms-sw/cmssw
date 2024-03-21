import FWCore.ParameterSet.Config as cms

from ..modules.hltRpcRecHits_cfi import *
from ..sequences.csclocalrecoSequence_cfi import *
from ..sequences.dtlocalrecoSequence_cfi import *
from ..sequences.gemLocalRecoSequence_cfi import *

muonlocalrecoSequence = cms.Sequence(csclocalrecoSequence+dtlocalrecoSequence+gemLocalRecoSequence+hltRpcRecHits)
