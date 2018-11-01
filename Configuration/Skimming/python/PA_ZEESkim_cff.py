import FWCore.ParameterSet.Config as cms


# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZEEPA = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZEEPA.HLTPaths = ["HLT_PADoublePhoton15_Eta3p1_Mass50_1000_v*"]
hltZEEPA.throw = False
hltZEEPA.andOr = True

# selection of valid vertex
primaryVertexFilterForZEEPA = cms.EDFilter("VertexSelector",
                                         src = cms.InputTag("offlinePrimaryVertices"),
                                         cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
                                         filter = cms.bool(True),   # otherwise it won't filter the events
                                         )

# single lepton selector
goodElectronsForZEEPA = cms.EDFilter("GsfElectronRefSelector",
                                   src = cms.InputTag("gedGsfElectrons"),
                                   cut = cms.string("pt > 25 && abs(eta)<1.44")
                                   )

## dilepton selectors
diElectronsForZEEPA = cms.EDProducer("CandViewShallowCloneCombiner",
                                   decay       = cms.string("goodElectronsForZEEPA goodElectronsForZEEPA"),
                                   checkCharge = cms.bool(False),
                                   cut         = cms.string("mass > 80 && mass < 110")
                                   )

# dilepton counter
diElectronsFilterForZEEPA = cms.EDFilter("CandViewCountFilter",
                                       src = cms.InputTag("diElectronsForZEEPA"),
                                       minNumber = cms.uint32(1)
                                       )

# Z->ee skim sequence
zEEPASkimSequence = cms.Sequence(
    hltZEEPA *
    primaryVertexFilterForZEEPA *
    goodElectronsForZEEPA * 
    diElectronsForZEEPA * 
    diElectronsFilterForZEEPA
)
