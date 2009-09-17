import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.stringResolutionProvider_cfi import *

electronResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                       resolutions     = ['et * (sqrt(5.6*5.6/(et*et) + 1.25/et + 0.033))', # add sigma(Et) not sigma(Et)/Et here
                                                          '0.03  + 1.0/et',                                 # add sigma(eta) here
                                                          '0.015 + 1.5/et'                                  # add sigma(phi) here
                                                          ],
                                       constraints     =  cms.vdouble(0)                                    # add constraints here
                                       )

muonResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                       resolutions     = ['et * (sqrt(5.6*5.6/(et*et) + 1.25/et + 0.033))',
                                                          '0.03  + 1.0/et',
                                                          '0.015 + 1.5/et'
                                                          ],
                                       constraints     = cms.vdouble(0)                                        
                                       )

tauResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                       resolutions     = ['et * (sqrt(5.6*5.6/(et*et) + 1.25/et + 0.033))',
                                                          '0.03  + 1.0/et',
                                                          '0.015 + 1.5/et'
                                                          ],
                                       constraints     =  cms.vdouble(0)
                                       )

jetResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                       resolutions     = ['et * (sqrt(5.6*5.6/(et*et) + 1.25/et + 0.033))',
                                                          '0.03  + 1.0/et',
                                                          '0.015 + 1.5/et'
                                                          ],
                                       constraints     =  cms.vdouble(0)
                                       )

metResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                       resolutions     = ['et * (sqrt(5.6*5.6/(et*et) + 1.25/et + 0.033))',
                                                          '0.03  + 1.0/et',
                                                          '0.015 + 1.5/et'
                                                          ],
                                       constraints     =  cms.vdouble(0)
                                       )
