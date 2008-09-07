#Default Pythia Paramters for Hydjet & Pyquen

import FWCore.ParameterSet.Config as cms

pyquenPythiaDefaultBlock = cms.PSet(
    
    pythiaDefault = cms.vstring('MSEL=1', # ! hard QCD on',
                                'MSTU(21)=1', # ! to avoid stopping run',
                                'PARU(14)=1.', # ! tolerance parameter to adjust fragmentation',
                                'MSTP(81)=0', # ! pp multiple scattering off',
                                'PMAS(5,1)=4.8', # ! b quark mass',
                                'PMAS(6,1)=175.0', # ! t quark mass'
                                'CKIN(3)=7.',# ! ptMin
                                'MSTJ(22)=2',
                                'PARJ(71)=10.' # Decays only if life time < 10mm
                                ),
    pythiaJets = cms.vstring('MSUB(11)=1', # q+q->q+q
                             'MSUB(12)=1', # q+qbar->q+qbar
                             'MSUB(13)=1', # q+qbar->g+g
                             'MSUB(28)=1', # q+g->q+g
                             'MSUB(53)=1', # g+g->q+qbar
                             'MSUB(68)=1' # g+g->g+g
                             ),
    pythiaPromptPhotons = cms.vstring('MSUB(14)=1', # q+qbar->g+gamma
                                      'MSUB(18)=1', # q+qbar->gamma+gamma
                                      'MSUB(29)=1', # q+g->q+gamma
                                      'MSUB(114)=1', # g+g->gamma+gamma
                                      'MSUB(115)=1' # g+g->g+gamma
                                      ),
    csa08Settings = cms.vstring('MSEL=0', # ! Only user defined processes,
                                'PARP(67)=1.',
                                'PARP(82)=1.9',
                                'PARP(85)=0.33',
                                'PARP(86)=0.66',
                                'PARP(89)=1000.',
                                'PARP(91)=1.0',
                                'MSTJ(11)=3',
                                'MSTJ(22)=2'
                                ),
    pythiaMuonCandidates = cms.vstring(
    'CKIN(3)=20',
    'MSTJ(22)=2',
    'PARJ(71)=40.'
    ),
    myParameters = cms.vstring(
    )
    
    )

