import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol03CfgV2

heepIDVarValueMaps = cms.EDProducer("ElectronHEEPIDValueMapProducer",
                                    beamSpot=cms.InputTag("offlineBeamSpot"),
                                    ebRecHitsAOD=cms.InputTag("reducedEcalRecHitsEB"),
                                    eeRecHitsAOD=cms.InputTag("reducedEcalRecHitsEB"),
                                    candsAOD=cms.VInputTag("packedCandsForTkIso",
                                                           "lostTracksForTkIso",
                                                           "lostTracksForTkIso:eleTracks"),
                                    #because GsfTracks of electrons are in "packedPFCandidates" 
                                    #end KF tracks of electrons are in lostTracks:eleTracks, need to
                                    #tell producer to veto electrons in the first collection
                                    candVetosAOD=cms.vstring("ELES","NONE","NONELES"),
                                    elesAOD=cms.InputTag("gedGsfElectrons"),
                                    ebRecHitsMiniAOD=cms.InputTag("reducedEgamma","reducedEBRecHits"),
                                    eeRecHitsMiniAOD=cms.InputTag("reducedEgamma","reducedEERecHits"),
                                    candsMiniAOD=cms.VInputTag("packedPFCandidates",
                                                               "lostTracks",
                                                               "lostTracks:eleTracks"),
                                    candVetosMiniAOD=cms.vstring("ELES","NONE","NONELES"),
                                    elesMiniAOD=cms.InputTag("slimmedElectrons"),
                                    dataFormat=cms.int32(0),#0 = auto detection, 1 = AOD, 2 = miniAOD
                                    trkIsoConfig= trkIsol03CfgV2
                                    )
