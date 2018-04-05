import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.Isolation.isoDeposits_cfi import isoDeposits as _isoDeposits



def isoDepositReplace( toBeIsolated, isolating):
    newDepositProducer = _isoDeposits.clone()
    newDepositProducer.src = toBeIsolated
    newDepositProducer.ExtractorPSet.inputCandView = isolating
    return newDepositProducer

#def candIsolatorReplace( isoDepositsSource ):
#    newCandIsolator = candIsolatorFromDeposits.clone()
#    newCandIsolator.deposits.src = isoDepositsSource
#    return newCandIsolator
