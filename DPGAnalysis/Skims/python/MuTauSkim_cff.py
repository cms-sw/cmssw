import FWCore.ParameterSet.Config as cms


'''
## 2012 HLT COND.S AS GIVEN BY ARUN IN 53X
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel as hlt_selector
hlt_selector.throw = cms.bool(False)
TauSkimMuTauMETHLT = hlt_selector.clone()
TauSkimMuTauMETHLT.TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT')
TauSkimMuTauMETHLT.andOr=cms.bool(True)
TauSkimMuTauMETHLT.HLTPaths = cms.vstring("HLT_IsoMu15_eta2p1_L1ETM20_v*", "HLT_IsoMu15Rho_eta2p1_L1ETM20_v*")
'''

TauSkimPFTausSelected = cms.EDFilter("PFTauSelector",
  src = cms.InputTag("hpsPFTauProducer"),
  discriminators = cms.VPSet(
  cms.PSet(  #discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),      #53X AND 75X
             discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"), #HTT 2015 TWIKI  
             selectionCut=cms.double(0.5)
      ),

  cms.PSet( #discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr"), ## 53X
            #discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseIsolation"),   #75X
            discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"), #HTT 2015 TWIKI   
            selectionCut=cms.double(0.5)
      ),


  ),
  #cut = cms.string('pt > 22. && abs(eta) < 2.3') #53X
  #cut = cms.string('et > 15. && abs(eta) < 2.5')  #75X
  cut = cms.string('pt > 18. && abs(eta) < 2.3') #HTT 2015 TWIKI
)

TauSkimPFTauSkimmedBy1 = cms.EDFilter("CandViewCountFilter",
 src = cms.InputTag('TauSkimPFTausSelected'),
 minNumber = cms.uint32(1)
)

TauSkimPFTauSkimmedBy2 = cms.EDFilter("CandViewCountFilter",
 src = cms.InputTag('TauSkimPFTausSelected'),
 minNumber = cms.uint32(2)
)

## MODULE IN 53X ONLY
TauSkimDiTauPairs = cms.EDProducer("CandViewShallowCloneCombiner",
                                   decay = cms.string("TauSkimPFTausSelected TauSkimPFTausSelected"),
                                   checkCharge = cms.bool(False),
                                   cut = cms.string("sqrt((daughter(0).eta-daughter(1).eta)*(daughter(0).eta-daughter(1).eta)+ min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi) ) * min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi) ) )>0.5"),
                                   )

## MODULE IN 53X ONLY  
TauSkimDiTauPairFilter = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("TauSkimDiTauPairs"),
                                      minNumber = cms.uint32(1)
                                      )


## MODULE IN 53X ONLY  
TauSkimPFTausSelectedForMuTau = TauSkimPFTausSelected.clone()
TauSkimPFTausSelectedForMuTau.discriminators = cms.VPSet(
    cms.PSet( #discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"), # 53X AND 75X
              discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"), #HTT 2015 TWIKI   
              selectionCut=cms.double(0.5)
              ),
    cms.PSet( #discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr"), #53X
              #discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseIsolation"),   #75X
              discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"), #HTT 2015 TWIKI 
              selectionCut=cms.double(0.5)
              ),
    cms.PSet( #discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection"), #53X
              discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"), #HTT 2015 TWIKI 
              selectionCut=cms.double(0.5)
              ),
    #cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByElectronVLooseMVA5"), #HTT 2015 TWIKI (not working!)
    #          selectionCut=cms.double(0.5)
    #          ),


    )

## MODULE IN 53X ONLY  
#TauSkimPFTausSelectedForMuTau.cut = cms.string('pt > 18. && abs(eta) < 2.3') #75X
TauSkimPFTausSelectedForMuTau.cut = cms.string('pt > 18. && abs(eta) < 2.3') #HTT 2015 TWIKI   
TauSkimPFTauSkimmedForMuTau = cms.EDFilter("CandViewCountFilter",
                                            src = cms.InputTag('TauSkimPFTausSelectedForMuTau'),
                                            minNumber = cms.uint32(1)
                                           )
## NO MuTauMET MODULES IMPLEMETED HERE FROM 53X


## MODULE IN 53X ONLY 
TauSkimMuonSelected = cms.EDFilter("MuonRefSelector",
                                   src = cms.InputTag("muons"),
                                   cut = cms.string("pt > 20 && " + "abs(eta) < 2.1 && " +
                                                    "isGlobalMuon && isTrackerMuon" +
                                                    #" && globalTrack.isNonnull "+
                                                    #" && globalTrack.hitPattern.numberOfValidTrackerHits>=5"+
                                                    #" && globalTrack.normalizedChi2<20"+
                                                    " && (pfIsolationR03.sumChargedHadronPt/pt) < 0.3"
                                                    ),
                                     )

## MODULE IN 53X ONLY   
TauSkimMuonSkimmedBy1 = cms.EDFilter("CandViewCountFilter",
                                     src = cms.InputTag('TauSkimMuonSelected'),
                                     minNumber = cms.uint32(1)
                                     )

## MODULE IN 53X ONLY   
TauSkimMuTauPairs = cms.EDProducer("CandViewShallowCloneCombiner",
                                    decay = cms.string("TauSkimMuonSelected TauSkimPFTausSelectedForMuTau"),
                                    checkCharge = cms.bool(False),
                                    cut         = cms.string("sqrt((daughter(0).eta-daughter(1).eta)*(daughter(0).eta-daughter(1).eta)+ min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi) ) * min( abs(daughter(0).phi-daughter(1).phi), 2*3.1415926 - abs(daughter(0).phi-daughter(1).phi) ) )>0.3"), ## DR CUT LOOSENED FOLLOWING MICHAL'S SUGGESTIONS FROM 0.5 TO 0.3
                                    )


## NO MuTauMET MODULES IMPLEMETED HERE FROM 53X

## MODULE IN 53X ONLY   
TauSkimMuTauPairFilter = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("TauSkimMuTauPairs"),
                                      minNumber = cms.uint32(1)
                                      )

## NO MuTauMET AND DITAU SEQUENCES IMPLEMETED HERE FROM 53X

## MUTAU SEQUENCE IN 53X ONLY   
mutauSkimSequence = cms.Sequence(
     TauSkimMuonSelected *
     TauSkimMuonSkimmedBy1 *
     TauSkimPFTausSelectedForMuTau *
     TauSkimPFTauSkimmedForMuTau *
     TauSkimMuTauPairs *
     TauSkimMuTauPairFilter
     )





