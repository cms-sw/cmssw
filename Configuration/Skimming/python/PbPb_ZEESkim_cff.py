import FWCore.ParameterSet.Config as cms


# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZEEHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZEEHI.HLTPaths = ["HLT_HIDoubleEle10Gsf_v*"]
hltZEEHI.throw = False
hltZEEHI.andOr = True

# selection of valid vertex
primaryVertexFilterForZEE = cms.EDFilter("VertexSelector",
                                         src = cms.InputTag("offlinePrimaryVertices"),
                                         cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
                                         filter = cms.bool(True),   # otherwise it won't filter the events
                                         )

# single lepton selector
goodElectronsForZEE = cms.EDFilter("GsfElectronRefSelector",
                                   src = cms.InputTag("gedGsfElectrons"),
                                   cut = cms.string("pt > 25 && abs(eta)<1.44")
                                   )

## dilepton selectors
diElectronsForZEE = cms.EDProducer("CandViewShallowCloneCombiner",
                                   decay       = cms.string("goodElectronsForZEE goodElectronsForZEE"),
                                   checkCharge = cms.bool(False),
                                   cut         = cms.string("mass > 80 && mass < 110")
                                   )

# dilepton counter
diElectronsFilterForZEE = cms.EDFilter("CandViewCountFilter",
                                       src = cms.InputTag("diElectronsForZEE"),
                                       minNumber = cms.uint32(1)
                                       )

# Z->ee skim sequence
zEESkimSequence = cms.Sequence(
    hltZEEHI *
    primaryVertexFilterForZEE *
    goodElectronsForZEE * 
    diElectronsForZEE * 
    diElectronsFilterForZEE
)
