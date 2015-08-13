import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1NtupleProducer_cfi import *
from L1Trigger.L1TNtuples.l1ExtraTreeProducer_cfi import *
from L1Trigger.L1TNtuples.l1MenuTreeProducer_cfi import *

L1NtupleRAW = cms.Sequence(
  l1NtupleProducer
  +l1ExtraTreeProducer
  +l1MenuTreeProducer
)
