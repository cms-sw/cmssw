import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1NtupleProducer_cfi import *
from L1Trigger.L1TNtuples.l1ExtraTreeProducer_cfi import *
from L1Trigger.L1TNtuples.l1MenuTreeProducer_cfi import *
from L1Trigger.L1TNtuples.l1JetRecoTreeProducer_cfi import *
from L1Trigger.L1TNtuples.l1MuonRecoTreeProducer_cfi import *

L1NtupleAODRAW = cms.Sequence(
  l1NtupleProducer
  +l1ExtraTreeProducer
  +l1MenuTreeProducer
  +l1MenuTreeProducer
  +l1JetRecoTreeProducer
  #+l1MuonRecoTreeProducer                                                     
)

