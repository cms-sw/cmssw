import FWCore.ParameterSet.Config as cms

### Diphoton CS                                                                                                                       
photonCandsPt30HOverE01 = cms.EDFilter("PhotonRefSelector",  
                                       src = cms.InputTag("photons"),
                                       cut  = cms.string("(pt > 30) && (hadronicOverEm < 0.1)"),
                                       )

twoPhotonsPt30HOverE01 = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("photonCandsPt30HOverE01"),
                                      minNumber = cms.uint32(2),
                                      )

diphotonSkimSequence = cms.Sequence(
    photonCandsPt30HOverE01 *
    twoPhotonsPt30HOverE01
    )
