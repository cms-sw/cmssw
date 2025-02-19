import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
# Input source
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),

    # cross section: 20-30 == 326.7, 30-50 == 227.0, 50-80 == 93.17,
    # 80-120 == 31.48, 120-170 == 9.63, 170-230 == 2.92, 230-300 == 0.8852
#    crossSection = cms.untracked.double(691.7852),
    #
    # at 10 TeV it scales down to 426
    #
    crossSection = cms.untracked.double(425.6),
    comEnergy = cms.double(7000.0),
    PythiaParameters = cms.PSet(
    pythiaUESettingsBlock,

    processParameters = cms.vstring('MSEL=0       !User defined processes', 
                                    'MSUB(15)=1   ', 
                                    'MSUB(30)=1   ', 
                                    'MDME(174,1)=0     !Z decay into d dbar',
                                    'MDME(175,1)=0     !Z decay into u ubar',
                                    'MDME(176,1)=0     !Z decay into s sbar',
                                    'MDME(177,1)=0     !Z decay into c cbar',
                                    'MDME(178,1)=0     !Z decay into b bbar',
                                    'MDME(179,1)=0     !Z decay into t tbar',
                                    'MDME(182,1)=0     !Z decay into e- e+',
                                    'MDME(183,1)=0     !Z decay into nu_e nu_ebar',
                                    'MDME(184,1)=1     !Z decay into mu- mu+',
                                    'MDME(185,1)=0     !Z decay into nu_mu nu_mubar',
                                    'MDME(186,1)=0     !Z decay into tau- tau+',
                                    'MDME(187,1)=0     !Z decay into nu_tau nu_taubar',
                                    'CKIN(3)=20.          ! minimum pt hat for hard interactions', 
                                    'CKIN(4)=300.          ! maximum pt hat for hard interactions'),

    # This is a vector of ParameterSet names to be read, in this order
    parameterSets = cms.vstring('pythiaUESettings', 
                                'processParameters')
    
    )
)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Generator/python/ZmumuJets_Pt_20_300_GEN_7TeV_cfg.py,v $'),
    annotation = cms.untracked.string('ZmumuJets pt hat 20-300')
)
