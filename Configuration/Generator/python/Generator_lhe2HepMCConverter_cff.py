# wrapper to run lhe2HepMCConverter to directly convert lhe files into hepmc for validation.
# Added 11/18/2014 nugent@physik.rwth-aachen.de
import FWCore.ParameterSet.Config as cms

generator =  cms.EDProducer("LHE2HepMCConverter",
                            LHERunInfoProduct = cms.InputTag("generator"),
                            LHEEventProduct = cms.InputTag("externalLHEProducer")
                            )

