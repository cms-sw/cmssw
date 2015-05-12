import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltJetHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltJetHI.HLTPaths = ["HLT_HIJet50U"]
hltJetHI.throw = False
hltJetHI.andOr = True

# selection of valid vertex
#primaryVertexFilterForJets = cms.EDFilter("VertexSelector",
#    src = cms.InputTag("hiSelectedVertex"),
#    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
#    filter = cms.bool(True),   # otherwise it won't filter the events
#    )

from HeavyIonsAnalysis.Configuration.collisionEventSelection_cff import *


# jet energy correction (L2+L3) ??
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
icPu5CaloJetsL2L3 = cms.EDProducer('CaloJetCorrectionProducer',
    src = cms.InputTag('iterativeConePu5CaloJets'),
                                   correctors = cms.vstring('ic5CaloL2L3')
                                   )

# leading jet E_T filter
jetEtFilter = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("icPu5CaloJetsL2L3"),
    etMin = cms.double(60.0),
    minNumber = cms.uint32(1)
    )

# dijet E_T filter
#dijetEtFilter = cms.EDFilter("EtMinCaloJetCountFilter",
#    src = cms.InputTag("icPu5CaloJetsL2L3"),
#    etMin = cms.double(50.0),
#    minNumber = cms.uint32(2)
#    )

#from RecoHI.HiEgammaAlgos.hiEcalSpikeFilter_cfi import *
from HeavyIonsAnalysis.PhotonAnalysis.hiEcalRecHitSpikeFilter_cfi import *
hiEcalRecHitSpikeFilter.minEt = 20.0

#HCAL cleaning
#from JetMETAnalysis.HcalReflagging.hbherechitreflaggerJETMET_cfi import *
# Broken.. commented out by Yen-Jie
# Need to update

#hbheReflagNewTimeEnv = hbherechitreflaggerJETMET.clone()
#hbheReflagNewTimeEnv.timingshapedcutsParameters.hbheTimingFlagBit=cms.untracked.int32(8)

# HCAL Timing
hcalTimingFilter = cms.EDFilter("HcalTimingFilter",
                                        hbheHits = cms.untracked.InputTag("hbheReflagNewTimeEnv")
                                )


# hcal noise filter
from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import *
HBHENoiseFilter.minRatio = cms.double(-99999.0)
HBHENoiseFilter.maxRatio = cms.double(99999.0)
HBHENoiseFilter.minZeros = cms.int32(100)


from HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cff import *


# jet skim sequence
jetSkimSequence = cms.Sequence(hltJetHI
                               * collisionEventSelection
                               * icPu5CaloJetsL2L3
                               * jetEtFilter
                               #* dijetEtFilter
                               * hiEcalRecHitSpikeFilter
                               #* hbheReflagNewTimeEnv
                               * hcalTimingFilter
                               * HBHENoiseFilter
                               )

