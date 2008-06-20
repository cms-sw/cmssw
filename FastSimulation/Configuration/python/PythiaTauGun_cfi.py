import FWCore.ParameterSet.Config as cms

source = cms.Source(
    "PythiaSource",
    pythiaVerbosity = cms.untracked.bool(False),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(-15),
    DoubleParticle = cms.untracked.bool(True),
    Ptmin = cms.untracked.double(20.0),
    Ptmax = cms.untracked.double(420.0),
#    Emin = cms.untracked.double(10.0),
#    Emax = cms.untracked.double(10.0),
    Etamin = cms.untracked.double(0.0),
    Etamax = cms.untracked.double(2.4),
    Phimin = cms.untracked.double(0.0),
    Phimax = cms.untracked.double(360.0),
    PythiaParameters = cms.PSet(
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
            'pythiaTauJets'
        ),

        # Tau jets only
        pythiaTauJets = cms.vstring(
            'MDME(89,1)=0      ! no tau->electron', 
            'MDME(90,1)=0      ! no tau->muon'
        )
        
    )
    
)


