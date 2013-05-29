import FWCore.ParameterSet.Config as cms

### Monophoton CS

photonCandsPt130HOverE05Barrel = cms.EDFilter("PhotonRefSelector",  
                                       src = cms.InputTag("photons"),
                                       cut  = cms.string("(pt > 130) && (hadTowOverEm < 0.05) && (isEB == 1)"),
                                       )

onePhotonPt130HOverE05 = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("photonCandsPt130HOverE05Barrel"),
                                      minNumber = cms.uint32(1),
                                      )
pfMETSelectorMono = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("pfMet"),
    cut = cms.string( "pt()>120" )
    )

pfMETCounterMono = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("pfMETSelectorMono"),
    minNumber = cms.uint32(1),
    )
monophotonSkimSequence = cms.Sequence(
    photonCandsPt130HOverE05Barrel *
    onePhotonPt130HOverE05 *
    pfMETSelectorMono *
    pfMETCounterMono
    )
