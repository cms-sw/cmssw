import FWCore.ParameterSet.Config as cms


from RecoMuon.MuonIsolation.muonPFIsolationDeposits_cff import *
from RecoMuon.MuonIsolation.muonPFIsolationValues_cff import *

muonPFIsolationTask =  cms.Task(
    muonPFIsolationDepositsTask,
    muonPFIsolationValuesTask
)                                         
muonPFIsolationSequence = cms.Sequence(muonPFIsolationTask)

from RecoMuon.MuonIsolation.displacedMuonPFIsolationDeposits_cff import *
from RecoMuon.MuonIsolation.displacedMuonPFIsolationValues_cff import *

displacedMuonPFIsolationTask =  cms.Task(
    displacedMuonPFIsolationDepositsTask,
    displacedMuonPFIsolationValuesTask
)
displacedMuonPFIsolationSequence = cms.Sequence(displacedMuonPFIsolationTask)
