import FWCore.ParameterSet.Config as cms

energyOverMomentumTree = cms.EDAnalyzer('EopElecTreeWriter',
                                        src  = cms.InputTag('electronGsfTracks'),
                                        # HLT path to filter on
                                        triggerPath = cms.string("HLT_Ele"), # ("HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v4"),
                                        # take HLT objects from this filter
                                        hltFilter   = cms.string("hltDiEle27L1DoubleEGWPTightHcalIsoFilter"),
                                        # debug the trigger and filter selections
                                        debugTriggerSelection = cms.bool(False),
                                        # Lower threshold on track's momentum magnitude in GeV
                                        #PCut = cms.double(31.),#15.),
                                        # tag on Z mass width window, between MzMin and MzMax
                                        #MzMin = cms.double(60.),
                                        #MzMax = cms.double(120.),
                                        # Lower threshold on Ecal energy deposit considering the Energy of SuperCluster
                                        #EcalEnergyCut = cms.double(30.),#14.),
                                        # Upper threshold on Hcal energy deposit considering the Energy of 5x5 Calo cells
                                        #HcalEnergyCut = cms.double(3.),
                                        # Isolation criteria: no other track lying in a cone of differential
                                        # radius dRIso (in eta-phi plan) arround the considered track
                                        #dRIso = cms.double(0.1),
                                        # Isolation against neutral particles:
                                        # SCdRMatch: differential radius (eta-phi plan) used for track-SupClus matching
                                        # SCdRIso: between SCdRMatch and SCdRIso arround track, NO OTHER Super Cluster
                                        #SCdRMatch = cms.double(0.09),
                                        #SCdRIso = cms.double(0.2)
                                        )
