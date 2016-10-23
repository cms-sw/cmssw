import FWCore.ParameterSet.Config as cms

#cuts
ELECTRON_CUT=("pt > 25 && abs(eta)<1.44")
DIELECTRON_CUT=("mass > 80 && mass < 110")

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZEEHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZEEHI.HLTPaths = ["HLT_PADoublePhoton15_Eta3p1_Mass50_1000_v*"]
hltZEEHI.throw = False
hltZEEHI.andOr = True

# selection of valid vertex
primaryVertexFilterForZEE = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# single lepton selector
goodElectrons = cms.EDFilter("GsfElectronRefSelector",
                                src = cms.InputTag("gedGsfElectrons"),
                                cut = cms.string(ELECTRON_CUT)
)

# dilepton selectors
diElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                             decay       = cms.string("goodElectrons goodElectrons"),
                             checkCharge = cms.bool(False),
                             cut         = cms.string(DIELECTRON_CUT)
)

# dilepton counter
diElectronsFilter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("diElectrons"),
                                    minNumber = cms.uint32(1)
)

# Z->ee skim sequence
zEESkimSequence = cms.Sequence(
    hltZEEHI *
    primaryVertexFilterForZEE *
    goodElectrons * 
    diElectrons * 
    diElectronsFilter
)
