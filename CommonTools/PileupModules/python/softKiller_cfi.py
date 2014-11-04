import FWCore.ParameterSet.Config as cms


softKiller = cms.EDProducer("SoftKillerProducer",
                       PFCandidates       = cms.string('particleFlow'),
                       Rho_EtaMax = cms.double( 5.0 ),
                       rParam = cms.double( 0.4 )
)
