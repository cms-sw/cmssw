import FWCore.ParameterSet.Config as cms

# Back to back hadronic taus
source = cms.Source("PythiaSource",
    Phimin = cms.untracked.double(0.0),
    maxEventsToPrint = cms.untracked.int32(5),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(15),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    Etamin = cms.untracked.double(0.0),
    DoubleParticle = cms.untracked.bool(True),
    Phimax = cms.untracked.double(360.0),
    Ptmin = cms.untracked.double(15.0),
    Ptmax = cms.untracked.double(60.0),
    Etamax = cms.untracked.double(2.5),
    PythiaParameters = cms.PSet(
#        process.pythiaUESettingsBlock,
        tauHadronicalOnly = cms.vstring('MDME( 89, 1) = 0    !no tau decay into electron', 'MDME( 90,1) = 0 !no tau decay into muon'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('tauHadronicalOnly')
    )
)

