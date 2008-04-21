import FWCore.ParameterSet.Config as cms

source = cms.Source("PythiaSource",
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        pythiaZtt = cms.vstring('MSEL = 11 ', 
            'MDME( 174,1) = 0  !Z decay into d dbar', 
            'MDME( 175,1) = 0  !Z decay into u ubar', 
            'MDME( 176,1) = 0  !Z decay into s sbar', 
            'MDME( 177,1) = 0  !Z decay into c cbar', 
            'MDME( 178,1) = 0  !Z decay into b bbar', 
            'MDME( 179,1) = 0  !Z decay into t tbar', 
            'MDME( 182,1) = 0  !Z decay into e- e+', 
            'MDME( 183,1) = 0  !Z decay into nu_e nu_ebar', 
            'MDME( 184,1) = 0  !Z decay into mu- mu+', 
            'MDME( 185,1) = 0  !Z decay into nu_mu nu_mubar', 
            'MDME( 186,1) = 1  !Z decay into tau- tau+', 
            'MDME( 187,1) = 0  !Z decay into nu_tau nu_taubar', 
            'MSTJ( 11) = 3     !Choice of the fragmentation function', 
            'MSTP( 2) = 1      !which order running alphaS', 
            'MSTP( 33) = 0     !(D=0) ', 
            'MSTP( 51) = 7     !structure function chosen', 
            'MSTP( 81) = 1     !multiple parton interactions 1 is Pythia default', 
            'MSTP( 82) = 4     !Defines the multi-parton model', 
            'PARJ( 71) = 10.   !for which ctau  10 mm', 
            'PARP( 82) = 1.9   !pt cutoff for multiparton interactions', 
            'PARP( 89) = 1000. !sqrts for which PARP82 is set', 
            'PARP( 83) = 0.5   !Multiple interactions: matter distrbn parameter', 
            'PARP( 84) = 0.4   !Multiple interactions: matter distribution parameter', 
            'PARP( 90) = 0.16  !Multiple interactions: rescaling power', 
            'CKIN( 1) = 40.    !(D=2. GeV)', 
            'CKIN( 2) = -1.    !(D=-1. GeV)', 
            'MDME(89,1)=0      ! no tau->electron', 
            'MDME(90,1)=0      ! no tau->muon'),
        parameterSets = cms.vstring('pythiaZtt')
    )
)


