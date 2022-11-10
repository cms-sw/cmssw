import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

DiMuonMassBiasClient = DQMEDHarvester("DiMuonMassBiasClient",
                                      FolderName = cms.string('DiMuonMassBiasMonitor'),
                                      fitBackground = cms.bool(False),
                                      useRooCBShape = cms.bool(False),
                                      useRooCMSShape = cms.bool(False),
                                      debugMode = cms.bool(False),
                                      fit_par = cms.PSet(
                                          mean_par = cms.vdouble(
                                              90.,
                                              60.,
                                              120.
                                          ),
                                          width_par = cms.vdouble(
                                              5.0,
                                              0.0,
                                              120.0
                                          ),
                                          sigma_par = cms.vdouble(
                                              5.0,
                                              0.0,
                                              120.0
                                          )
                                      ),
                                      MEtoHarvest = cms.vstring(
                                          'DiMuMassVsMuMuPhi',
                                          'DiMuMassVsMuMuEta',
                                          'DiMuMassVsMuPlusPhi',
                                          'DiMuMassVsMuPlusEta',
                                          'DiMuMassVsMuMinusPhi',
                                          'DiMuMassVsMuMinusEta',
                                          'DiMuMassVsMuMuDeltaEta',
                                          'DiMuMassVsCosThetaCS'
                                      )
                                  )
