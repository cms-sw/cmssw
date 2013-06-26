import FWCore.ParameterSet.Config as cms

""" 
######## Generate Z->tautau decays ###########
Require both taus to decay hadronically 
"""

source = cms.Source("PythiaSource",
    PythiaParameters = cms.PSet(
        #
        # Default cards for minimum bias events (unfiltered)
        # Name of the set is "pythiaMinBias"
        #include "IOMC/GeneratorInterface/test/pythiaMinBias.cfg"
        #
        # User cards - name is "myParameters"
        # Pythia's random generator initialization 
        zToTauTauHadronicOnly = cms.vstring('MSEL         = 11 ',           
             'MDME( 174,1) = 0    !Z decay into d dbar',
             'MDME( 175,1) = 0    !Z decay into u ubar',
             'MDME( 176,1) = 0    !Z decay into s sbar',
             'MDME( 177,1) = 0    !Z decay into c cbar',
             'MDME( 178,1) = 0    !Z decay into b bbar',
             'MDME( 179,1) = 0    !Z decay into t tbar',
             'MDME( 182,1) = 0    !Z decay into e- e+',
             'MDME( 183,1) = 0    !Z decay into nu_e nu_ebar',
             'MDME( 184,1) = 0    !Z decay into mu- mu+',
             'MDME( 185,1) = 0    !Z decay into nu_mu nu_mubar',
             'MDME( 186,1) = 1    !Z decay into tau- tau+',
             'MDME( 187,1) = 0    !Z decay into nu_tau nu_taubar', 
             'MDME( 89, 1) = 0    !no tau decay into electron',
             'MDME( 90, 1) = 0    !no tau decay into muon',
             'CKIN( 1)     = 40.  !(D=2. GeV)',
             'CKIN( 2)     = -1.  !(D=-1. GeV)'),
        parameterSets = cms.vstring('zToTauTauHadronicOnly')
    )
)
