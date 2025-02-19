import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel as hlt_selector
hlt_selector.throw = cms.bool(False)
TauSkimMuTauMETHLT = hlt_selector.clone()
TauSkimMuTauMETHLT.TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT')
TauSkimMuTauMETHLT.andOr=cms.bool(True)
TauSkimMuTauMETHLT.HLTPaths = cms.vstring("HLT_IsoMu15_eta2p1_L1ETM20_v*", "HLT_IsoMu15Rho_eta2p1_L1ETM20_v*")

TauSkimPFTausSelected = cms.EDFilter("PFTauSelector",
   src = cms.InputTag("hpsPFTauProducer"),
   discriminators = cms.VPSet(
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
		   selectionCut=cms.double(0.5)           
	),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr"),
		   selectionCut=cms.double(0.5)           
	),
   ),
   cut = cms.string('pt > 22. && abs(eta) < 2.3') 
)

TauSkimPFTauSkimmedBy1 = cms.EDFilter("CandViewCountFilter",
 src = cms.InputTag('TauSkimPFTausSelected'),
 minNumber = cms.uint32(1)
)

TauSkimPFTauSkimmedBy2 = cms.EDFilter("CandViewCountFilter",
 src = cms.InputTag('TauSkimPFTausSelected'),
 minNumber = cms.uint32(2)
)

TauSkimDiTauPairs  = cms.EDProducer("CandViewShallowCloneCombiner",
                                    decay = cms.string("TauSkimPFTausSelected TauSkimPFTausSelected"),
                                    checkCharge = cms.bool(False),
                                    cut         = cms.string("sqrt((daughter(0).eta-daughter(1).eta)*(daughter(0).eta-daughter(1).eta)+  min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi)  ) *  min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi)  )  )>0.5"),
                                    )

TauSkimDiTauPairFilter = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("TauSkimDiTauPairs"),
                                      minNumber = cms.uint32(1)
                                      )

TauSkimPFTausSelectedForMuTau = TauSkimPFTausSelected.clone()
TauSkimPFTausSelectedForMuTau.discriminators = cms.VPSet(
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
              selectionCut=cms.double(0.5)
              ),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr"),
              selectionCut=cms.double(0.5)
              ),
    cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection"),
              selectionCut=cms.double(0.5)
              ),
    )
TauSkimPFTausSelectedForMuTau.cut = cms.string('pt > 18. && abs(eta) < 2.3')
TauSkimPFTauSkimmedForMuTau = cms.EDFilter("CandViewCountFilter",
                                           src = cms.InputTag('TauSkimPFTausSelectedForMuTau'),
                                           minNumber = cms.uint32(1)
                                           )

TauSkimPFTausSelectedForMuTauMET = TauSkimPFTausSelected.clone()
TauSkimPFTausSelectedForMuTauMET.cut = cms.string('pt > 18. && abs(eta) < 2.3')
TauSkimPFTauSkimmedForMuTauMET = cms.EDFilter("CandViewCountFilter",
                                              src = cms.InputTag('TauSkimPFTausSelectedForMuTauMET'),
                                              minNumber = cms.uint32(1)
                                              )

TauSkimMuonSelected = cms.EDFilter("MuonRefSelector",
                                   src = cms.InputTag("muons"),
                                   cut = cms.string("pt > 15 && " + "abs(eta) < 2.4 && " +
                                                    "isGlobalMuon && isTrackerMuon" +
                                                    " && globalTrack.isNonnull "+
                                                    " && globalTrack.hitPattern.numberOfValidTrackerHits>=5"+
                                                    " && globalTrack.normalizedChi2<20"+
                                                    " && (pfIsolationR03.sumChargedHadronPt/pt) < 0.3"
                                                    ),
                                   )

TauSkimMuonSkimmedBy1 = cms.EDFilter("CandViewCountFilter",
                                     src = cms.InputTag('TauSkimMuonSelected'),
                                     minNumber = cms.uint32(1)
                                     )

TauSkimMuTauPairs  = cms.EDProducer("CandViewShallowCloneCombiner",
                                    decay = cms.string("TauSkimMuonSelected TauSkimPFTausSelectedForMuTau"),
                                    checkCharge = cms.bool(False),
                                    cut         = cms.string("sqrt((daughter(0).eta-daughter(1).eta)*(daughter(0).eta-daughter(1).eta)+  min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi)  ) *  min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi)  )  )>0.5"),
                                    )

TauSkimMuTauMETPairs  = cms.EDProducer("CandViewShallowCloneCombiner",
                                       decay = cms.string("TauSkimMuonSelected TauSkimPFTausSelectedForMuTauMET"),
                                       checkCharge = cms.bool(False),
                                       cut         = cms.string("sqrt((daughter(0).eta-daughter(1).eta)*(daughter(0).eta-daughter(1).eta)+  min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi)  ) *  min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi)  )  )>0.5"),
                                       )

TauSkimMuTauPairFilter = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("TauSkimMuTauPairs"),
                                      minNumber = cms.uint32(1)
                                      )
TauSkimMuTauMETPairFilter = cms.EDFilter("CandViewCountFilter",
                                         src = cms.InputTag("TauSkimMuTauMETPairs"),
                                         minNumber = cms.uint32(1)
                                         )


tauSkim1Sequence = cms.Sequence(
    TauSkimPFTausSelected *
    TauSkimPFTauSkimmedBy1
    )

tauSkim2Sequence = cms.Sequence(
    TauSkimPFTausSelected *
    TauSkimPFTauSkimmedBy2 *
    TauSkimDiTauPairs *
    TauSkimDiTauPairFilter
    )

mutauSkimSequence = cms.Sequence(
    TauSkimMuonSelected *
    TauSkimMuonSkimmedBy1 *
    TauSkimPFTausSelectedForMuTau *
    TauSkimPFTauSkimmedForMuTau *
    TauSkimMuTauPairs *
    TauSkimMuTauPairFilter
    )

mutauMETSkimSequence = cms.Sequence(
    TauSkimMuTauMETHLT *
    TauSkimMuonSelected *
    TauSkimMuonSkimmedBy1 *
    TauSkimPFTausSelectedForMuTauMET *
    TauSkimPFTauSkimmedForMuTauMET *
    TauSkimMuTauMETPairs *
    TauSkimMuTauMETPairFilter
    )
