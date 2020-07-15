import FWCore.ParameterSet.Config as cms


TauSkimPFTausSelected = cms.EDFilter("PFTauSelector",
   src = cms.InputTag("hpsPFTauProducer"),
   discriminators = cms.VPSet(
      cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
         selectionCut=cms.double(0.5)           
      )
   ),
   discriminatorContainers = cms.VPSet(
      cms.PSet( discriminator=cms.InputTag("hpsPFTauBasicDiscriminators"),
         rawValues=cms.vstring(),
         selectionCuts=cms.vdouble(),
         workingPoints=cms.vstring("ByLooseCombinedIsolationDBSumPtCorr3Hits")
      )
   ),
   cut = cms.string('et > 15. && abs(eta) < 2.5') 
)

TauSkimPFTauSkimmedBy1 = cms.EDFilter("CandViewCountFilter",
 src = cms.InputTag('TauSkimPFTausSelected'),
 minNumber = cms.uint32(1)
)

TauSkimPFTauSkimmedBy2 = cms.EDFilter("CandViewCountFilter",
 src = cms.InputTag('TauSkimPFTausSelected'),
 minNumber = cms.uint32(2)
)


tauSkim1Sequence = cms.Sequence(
   TauSkimPFTausSelected *
   TauSkimPFTauSkimmedBy1
   )

tauSkim2Sequence = cms.Sequence(
   TauSkimPFTausSelected *
   TauSkimPFTauSkimmedBy2
   )

