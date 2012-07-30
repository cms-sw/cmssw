import FWCore.ParameterSet.Config as cms


from RecoMuon.MuonIsolation.muonPFIsolationDeposits_cff import *
from RecoMuon.MuonIsolation.muonPFIsolationValues_cff import *

muonPFIsolationSequence =  cms.Sequence(
    muonPFIsolationDepositsSequence + 
    muonPFIsolationValuesSequence
)                                         





                 

