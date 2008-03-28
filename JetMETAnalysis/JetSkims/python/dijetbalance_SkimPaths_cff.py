import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.dijetbalance_Sequences_cff import *
dijetbalance30HLTPath = cms.Path(dijetbalance30HLTFilter)
dijetbalance60HLTPath = cms.Path(dijetbalance60HLTFilter)
dijetbalance110HLTPath = cms.Path(dijetbalance110HLTFilter)
dijetbalance150HLTPath = cms.Path(dijetbalance150HLTFilter)
dijetbalance200HLTPath = cms.Path(dijetbalance200HLTFilter)

