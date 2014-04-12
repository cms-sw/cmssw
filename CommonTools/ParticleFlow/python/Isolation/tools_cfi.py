import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.Isolation.isoDeposits_cfi import *



def isoDepositReplace( toBeIsolated, isolating):
    newDepositProducer = isoDeposits.clone()
    newDepositProducer.src = toBeIsolated
    newDepositProducer.ExtractorPSet.inputCandView = isolating
    return newDepositProducer

#def candIsolatorReplace( isoDepositsSource ):
#    newCandIsolator = candIsolatorFromDeposits.clone()
#    newCandIsolator.deposits.src = isoDepositsSource
#    return newCandIsolator
