import FWCore.ParameterSet.Config as cms

pfcandAnalyzerCS = cms.EDAnalyzer('HiPFCandAnalyzer',
                                pfCandidateLabel = cms.InputTag("akCs4PFJets","pfParticlesCs"),
                                jetLabel = cms.InputTag("ak5patJets"),
                                genLabel = cms.InputTag("hiGenParticles"),
                                # debug
                                pfPtMin = cms.double(0.5),
                                genPtMin = cms.double(0.5),
                                jetPtMin = cms.double(20.0),                                
                                verbosity = cms.untracked.int32(0),
                                skipCharged = cms.untracked.bool(False),
                                doJets_ = cms.untracked.bool(False),
                                doVS = cms.untracked.bool(False),
                                bkg = cms.InputTag("voronoiBackgroundPF"),
                                etaBins = cms.int32(15),
                                fourierOrder = cms.int32(5),
                                doUEraw_ = cms.untracked.bool(True),                                
                                )

