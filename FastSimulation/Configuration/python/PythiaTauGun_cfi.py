import FWCore.ParameterSet.Config as cms

source = cms.Source("PythiaSource",
    Phimin = cms.untracked.double(0.0),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(-15),
    Etamin = cms.untracked.double(0.0),
    DoubleParticle = cms.untracked.bool(True),
    Phimax = cms.untracked.double(360.0),
    Ptmin = cms.untracked.double(20.0),
    Ptmax = cms.untracked.double(420.0),
    Etamax = cms.untracked.double(2.4),
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        # Tau jets only
        pythiaTauJets = cms.vstring('MDME(89,1)=0      ! no tau->electron', 
            'MDME(90,1)=0      ! no tau->muon'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaTauJets')
    )
)


