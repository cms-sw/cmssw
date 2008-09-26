import FWCore.ParameterSet.Config as cms

#
# track preselection
#
ueTracks = cms.EDFilter("TrackSelector",
                        src = cms.InputTag("selectTracks"),
                        cut = cms.string('pt > 0.89 & eta >= -2.0 & eta <= 2.0')
                        )

#
# jet preselection
#
ueSelectedJets = cms.EDProducer("EtaPtMinCandViewSelector",
                                src    = cms.InputTag("IC5TracksJet"),
                                ptMin  = cms.double( 0.89 ),
                                etaMin = cms.double( -2.0 ),
                                etaMax = cms.double(  2.0 )
                                )

ueLeadingJet = cms.EDProducer("LargestPtCandViewSelector",
                              src = cms.InputTag("ueSelectedJets"),
                              maxNumber = cms.uint32( 1 )
                              )

#
# region analysis
#
towardsTracks = cms.EDProducer('UERegionSelector',
                               JetCollectionName          = cms.untracked.InputTag("ueLeadingJet"),
                               TrackCollectionName        = cms.untracked.InputTag("ueTracks"),
                               DeltaPhiByPiMinJetParticle = cms.double( 0. ),
                               DeltaPhiByPiMaxJetParticle = cms.double( 1./3. )
                               )

transverseTracks = cms.EDProducer('UERegionSelector',
                                  JetCollectionName          = cms.untracked.InputTag("ueLeadingJet"),
                                  TrackCollectionName        = cms.untracked.InputTag("ueTracks"),
                                  DeltaPhiByPiMinJetParticle = cms.double( 1./3. ),
                                  DeltaPhiByPiMaxJetParticle = cms.double( 2./3. )
                                  )

awayTracks = cms.EDProducer('UERegionSelector',
                            JetCollectionName          = cms.untracked.InputTag("ueLeadingJet"),
                            TrackCollectionName        = cms.untracked.InputTag("ueTracks"),
                            DeltaPhiByPiMinJetParticle = cms.double( 2./3. ),
                            DeltaPhiByPiMaxJetParticle = cms.double( 1. )
                            )

UERegionSelector = cms.Sequence(ueTracks*ueSelectedJets*ueLeadingJet*towardsTracks*transverseTracks*awayTracks)
