import FWCore.ParameterSet.Config as cms

TauSpinnerReco = cms.EDProducer( "TauSpinnerCMS",
                                 isReco = cms.bool(True),
                                 isTauolaConfigured = cms.bool(False),
                                 isLHPDFConfigured = cms.bool(False),
                                 LHAPDFname = cms.untracked.string('MSTW2008nnlo90cl.LHgrid'),
                                 CMSEnergy = cms.double(8000.0),
                                 gensrc = cms.InputTag('genParticles')
                                 )

TauSpinnerGen = cms.EDProducer( "TauSpinnerCMS",
                                isReco = cms.bool(False),
                                isTauolaConfigured = cms.bool(True),
                                isLHPDFConfigured = cms.bool(True),
                                LHAPDFname = cms.untracked.string('MSTW2008nnlo90cl.LHgrid'),
                                CMSEnergy = cms.double(8000.0),
                                gensrc = cms.InputTag('generator'),
                                )
