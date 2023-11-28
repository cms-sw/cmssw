#Default Pythia Paramters for Hydjet & Pyquen

import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

collisionParameters4TeV = cms.PSet(aBeamTarget = cms.double(208.0), ## beam/target atomic number
                               comEnergy = cms.double(4000.0)
                               )

collisionParameters2760GeV = cms.PSet(aBeamTarget = cms.double(208.0), ## beam/target atomic number
                                   comEnergy = cms.double(2760.0)
                                   )

collisionParameters5362GeV = cms.PSet(aBeamTarget = cms.double(208.0), ## beam/target atomic number
                                      comEnergy = cms.double(5362.0)
                                  )

collisionParameters = collisionParameters2760GeV.clone()

qgpParameters = cms.PSet(qgpInitialTemperature = cms.double(1.0), ## initial temperature of QGP; allowed range [0.2,2.0]GeV;
                         qgpProperTimeFormation = cms.double(0.1), ## proper time of QGP formation; allowed range [0.01,10.0]fm/c;
                         hadronFreezoutTemperature = cms.double(0.14),
                         doRadiativeEnLoss = cms.bool(True), ## if true, perform partonic radiative en loss
                         doCollisionalEnLoss = cms.bool(False),
                         qgpNumQuarkFlavor = cms.int32(0),  ## number of active quark flavors in qgp; allowed values: 0,1,2,3 
                         )

pyquenParameters  = cms.PSet(doIsospin = cms.bool(True),
                             angularSpectrumSelector = cms.int32(0), ## angular emitted gluon spectrum :
                             embeddingMode = cms.int32(0),
                             )

hydjetParameters = cms.PSet(sigmaInelNN = cms.double(58),
                            shadowingSwitch = cms.int32(0),
                            nMultiplicity = cms.int32(21500),
                            fracSoftMultiplicity = cms.double(1.),
                            maxLongitudinalRapidity = cms.double(4.5),
                            maxTransverseRapidity = cms.double(1.),
                            rotateEventPlane = cms.bool(True),
                            allowEmptyEvents = cms.bool(False),
                            embeddingMode = cms.int32(0)                            
                            )

pyquenPythiaDefaultBlock = cms.PSet(
    pythiaUESettingsBlock,
    hydjetPythiaDefault = cms.vstring('MSEL=0   ! user processes',
                                      'CKIN(3)=6.',# ! ptMin
                                      'MSTP(81)=0'
                                      ),
    ppDefault = cms.vstring('MSEL=1   ! QCD hight pT processes (only jets)',
                            'CKIN(3)=6.',# ! ptMin
                            'MSTP(81)=0'
                            ),
    pythiaHirootDefault = cms.vstring('MSEL=0', # ! Only user defined processes,  
                                'MSTU(21)=1', # ! to avoid stopping run',
                                'PARU(14)=1.', # ! tolerance parameter to adjust fragmentation',
                                'MSTP(81)=0', # ! pp multiple scattering off',
                                'PMAS(5,1)=4.8', # ! b quark mass',
                                'PMAS(6,1)=175.0', # ! t quark mass'
                                'CKIN(3)=7.',# ! ptMin
                                'MSTJ(22)=2',
                                'PARJ(71)=10.', # Decays only if life time < 10mm
                                'PARP(67)=1.',
                                'PARP(82)=1.9',
                                'PARP(85)=0.33',
                                'PARP(86)=0.66',
                                'PARP(89)=1000.',
                                'PARP(91)=1.0',
                                'MSTJ(11)=3',
                                'MSTJ(22)=2'                                
                                ),
    ppJets = cms.vstring('MSEL=1   ! QCD hight pT processes'),
    customProcesses = cms.vstring('MSEL=0   ! User processes'),
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

    pythiaWeakBosons = cms.vstring('MSUB(1)=1',
                                   'MSUB(2)=1'),
    
    pythiaZjets = cms.vstring('MSUB(15)=1',
                              'MSUB(30)=1'),
    
    pythiaCharmoniumNRQCD = cms.vstring('MSUB(421) = 1',
                                        'MSUB(422) = 1',
                                        'MSUB(423) = 1',
                                        'MSUB(424) = 1',
                                        'MSUB(425) = 1',
                                        'MSUB(426) = 1',
                                        'MSUB(427) = 1',
                                        'MSUB(428) = 1',
                                        'MSUB(429) = 1',
                                        'MSUB(430) = 1',
                                        'MSUB(431) = 1',
                                        'MSUB(432) = 1',
                                        'MSUB(433) = 1',
                                        'MSUB(434) = 1',
                                        'MSUB(435) = 1',
                                        'MSUB(436) = 1',
                                        'MSUB(437) = 1',
                                        'MSUB(438) = 1',
                                        'MSUB(439) = 1'
                                        ),

    pythiaBottomoniumNRQCD = cms.vstring('MSUB(461) = 1',
                                         'MSUB(462) = 1',
                                         'MSUB(463) = 1',
                                         'MSUB(464) = 1',
                                         'MSUB(465) = 1',
                                         'MSUB(466) = 1',
                                         'MSUB(467) = 1',
                                         'MSUB(468) = 1',
                                         'MSUB(469) = 1',
                                         'MSUB(470) = 1',
                                         'MSUB(471) = 1',
                                         'MSUB(472) = 1',
                                         'MSUB(473) = 1',
                                         'MSUB(474) = 1',
                                         'MSUB(475) = 1',
                                         'MSUB(476) = 1',
                                         'MSUB(477) = 1',
                                         'MSUB(478) = 1',
                                         'MSUB(479) = 1',
                                         ),

    pythiaQuarkoniaSettings = cms.vstring('PARP(141)=1.16',   # Matrix Elements
                                          'PARP(142)=0.0119',
                                          'PARP(143)=0.01',
                                          'PARP(144)=0.01',
                                          'PARP(145)=0.05',
                                          'PARP(146)=9.28',
                                          'PARP(147)=0.15',
                                          'PARP(148)=0.02',
                                          'PARP(149)=0.02',
                                          'PARP(150)=0.085',                                                
                                          # Meson spin
                                          'PARJ(13)=0.60',
                                          'PARJ(14)=0.162',
                                          'PARJ(15)=0.018',
                                          'PARJ(16)=0.054',
                                          # Polarization
                                          'MSTP(145)=0',
                                          'MSTP(146)=0',
                                          'MSTP(147)=0',
                                          'MSTP(148)=1',
                                          'MSTP(149)=1',
                                          # Chi_c branching ratios
                                          'BRAT(861)=0.202',
                                          'BRAT(862)=0.798',
                                          'BRAT(1501)=0.013',
                                          'BRAT(1502)=0.987',
                                          'BRAT(1555)=0.356',
                                          'BRAT(1556)=0.644'
                                          ),
    
    pythiaZtoMuons = cms.vstring("MDME(174,1)=0",          # !Z decay into d dbar,
                                 "MDME(175,1)=0",          # !Z decay into u ubar,
                                 "MDME(176,1)=0",          # !Z decay into s sbar,
                                 "MDME(177,1)=0",          # !Z decay into c cbar,
                                 "MDME(178,1)=0",          # !Z decay into b bbar,
                                 "MDME(179,1)=0",          # !Z decay into t tbar,
                                 "MDME(182,1)=0",          # !Z decay into e- e+,
                                 "MDME(183,1)=0",          # !Z decay into nu_e nu_ebar,
                                 "MDME(184,1)=1",          # !Z decay into mu- mu+,
                                 "MDME(185,1)=0",          # !Z decay into nu_mu nu_mubar,
                                 "MDME(186,1)=0",          # !Z decay into tau- tau+,
                                 "MDME(187,1)=0"           # !Z decay into nu_tau nu_taubar
                                 ),


    pythiaZtoElectrons = cms.vstring("MDME(174,1)=0",          # !Z decay into d dbar,
                                     "MDME(175,1)=0",          # !Z decay into u ubar,
                                     "MDME(176,1)=0",          # !Z decay into s sbar,
                                     "MDME(177,1)=0",          # !Z decay into c cbar,
                                     "MDME(178,1)=0",          # !Z decay into b bbar,
                                     "MDME(179,1)=0",          # !Z decay into t tbar,
                                     "MDME(182,1)=1",          # !Z decay into e- e+,
                                     "MDME(183,1)=0",          # !Z decay into nu_e nu_ebar,
                                     "MDME(184,1)=0",          # !Z decay into mu- mu+,
                                     "MDME(185,1)=0",          # !Z decay into nu_mu nu_mubar,
                                     "MDME(186,1)=0",          # !Z decay into tau- tau+,
                                     "MDME(187,1)=0"           # !Z decay into nu_tau nu_taubar
                                     ),
    
    pythiaZtoMuonsAndElectrons = cms.vstring("MDME(174,1)=0",          # !Z decay into d dbar,
                                             "MDME(175,1)=0",          # !Z decay into u ubar,
                                             "MDME(176,1)=0",          # !Z decay into s sbar,
                                             "MDME(177,1)=0",          # !Z decay into c cbar,
                                             "MDME(178,1)=0",          # !Z decay into b bbar,
                                             "MDME(179,1)=0",          # !Z decay into t tbar,
                                             "MDME(182,1)=1",          # !Z decay into e- e+,
                                             "MDME(183,1)=0",          # !Z decay into nu_e nu_ebar,
                                             "MDME(184,1)=1",          # !Z decay into mu- mu+,
                                             "MDME(185,1)=0",          # !Z decay into nu_mu nu_mubar,
                                             "MDME(186,1)=0",          # !Z decay into tau- tau+,
                                             "MDME(187,1)=0"           # !Z decay into nu_tau nu_taubar
                                             ),
    
    pythiaUpsilonToMuons = cms.vstring('BRAT(1034) = 0 ',  # switch off',
                                       'BRAT(1035) = 1 ',  # switch on',
                                       'BRAT(1036) = 0 ',  # switch off',
                                       'BRAT(1037) = 0 ',  # switch off',
                                       'BRAT(1038) = 0 ',  # switch off',
                                       'BRAT(1039) = 0 ',  # switch off',
                                       'BRAT(1040) = 0 ',  # switch off',
                                       'BRAT(1041) = 0 ',  # switch off',
                                       'BRAT(1042) = 0 ',  # switch off',
                                       'MDME(1034,1) = 0 ',  # switch off',
                                       'MDME(1035,1) = 1 ',  # switch on',
                                       'MDME(1036,1) = 0 ',  # switch off',
                                       'MDME(1037,1) = 0 ',  # switch off',
                                       'MDME(1038,1) = 0 ',  # switch off',
                                       'MDME(1039,1) = 0 ',  # switch off',
                                       'MDME(1040,1) = 0 ',  # switch off',
                                       'MDME(1041,1) = 0 ',  # switch off',
                                       'MDME(1042,1) = 0 ',  # switch off'
                                       ),

   pythiaJpsiToMuons = cms.vstring('BRAT(858) = 0 ',  # switch off',
                                   'BRAT(859) = 1 ',  # switch on',
                                   'BRAT(860) = 0 ',  # switch off',
                                   'MDME(858,1) = 0 ',  # switch off',
                                   'MDME(859,1) = 1 ',  # switch on',
                                   'MDME(860,1) = 0 ',  # switch off'
                                   ),
    
    pythiaMuonCandidates = cms.vstring(
    'CKIN(3)=20',
    'MSTJ(22)=2',
    'PARJ(71)=40.'
    ),
    myParameters = cms.vstring(
    )
    
)    

# This one is not to be used
impactParameters = cms.PSet(cFlag = cms.int32(1),
                            bFixed = cms.double(0),
                            bMin = cms.double(0),
                            bMax = cms.double(30)
                            )

