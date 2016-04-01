import FWCore.ParameterSet.Config as cms
from RecoHI.HiJetAlgos.HiRecoJets_cff import *
from RecoHI.HiJetAlgos.HiRecoPFJets_cff import *
# from RecoJets.JetProducers.ak8PFJetsCS_cfi import ak8PFJetsCS
from RecoJets.JetProducers.akCs4PFJets_cfi import akCs4PFJets

# akCs4PFJets = ak8PFJetsCS.clone( 
    # src    = cms.InputTag('particleFlowTmp'),
    # rParam = cms.double(0.4),
    # jetPtMin = cms.double(0.0),
    # doAreaFastjet = cms.bool(True),
    # GhostArea = cms.double(0.005),
    # useConstituentSubtraction = cms.bool(False),
    # useConstituentSubtractionHi = cms.bool(True),
    # etaMap    = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
    # rho       = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoCorr'),
    # rhom      = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoMCorr'),
    # csAlpha   = cms.double(1.),
    # writeJetsWithConst = cms.bool(True),
    # verbosity = cms.int32(0),
    # jetCollInstanceName = cms.string("pfParticlesCs")
    
	##writeCompound = cms.bool(True)
    # )
	
akCs4PFJets.rho      = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoCorr1Bin')
akCs4PFJets.rhom      = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoMCorr1Bin')
akCs1PFJets = akCs4PFJets.clone(rParam       = cms.double(0.1))
akCs2PFJets = akCs4PFJets.clone(rParam       = cms.double(0.2))
akCs3PFJets = akCs4PFJets.clone(rParam       = cms.double(0.3))
akCs5PFJets = akCs4PFJets.clone(rParam       = cms.double(0.5))
akCs6PFJets = akCs4PFJets.clone(rParam       = cms.double(0.6))

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
akCsSoftDrop4PFJets = cms.EDProducer(
    "FastjetJetProducer",
    PFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.4),
    useSoftDrop = cms.bool(True),
    zcut = cms.double(0.1),
    beta = cms.double(0.0),
    R0   = cms.double(0.4),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
)
akCsSoftDrop4PFJets.src = cms.InputTag("akCs4PFJets","pfParticlesCs")
akCsSoftDrop5PFJets = akCsSoftDrop4PFJets.clone(rParam = cms.double(0.5),
                                                src    = cms.InputTag("akCs5PFJets","pfParticlesCs"))
akCsSoftDrop6PFJets = akCsSoftDrop4PFJets.clone(rParam = cms.double(0.6),
                                                src    = cms.InputTag("akCs6PFJets","pfParticlesCs"))

akCsFilter4PFJets = cms.EDProducer(
    "FastjetJetProducer",
    PFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.4),
    useFiltering = cms.bool(True),
    nFilt = cms.int32(4),
    rFilt = cms.double(0.15),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
)
akCsFilter4PFJets.src = cms.InputTag("akCs4PFJets","pfParticlesCs")
akCsFilter5PFJets = akCsFilter4PFJets.clone(rParam = cms.double(0.5),
                                            src    = cms.InputTag("akCs5PFJets","pfParticlesCs"))
akCsFilter6PFJets = akCsFilter4PFJets.clone(rParam = cms.double(0.6),
                                            src    = cms.InputTag("akCs6PFJets","pfParticlesCs"))

ak5PFJets.doAreaFastjet = cms.bool(True)

akVs1CaloJets.doAreaFastjet = cms.bool(True)
akVs2CaloJets.doAreaFastjet = cms.bool(True)
akVs3CaloJets.doAreaFastjet = cms.bool(True)
akVs4CaloJets.doAreaFastjet = cms.bool(True)
akVs5CaloJets.doAreaFastjet = cms.bool(True)
akVs6CaloJets.doAreaFastjet = cms.bool(True)

akVs1PFJets.doAreaFastjet = cms.bool(True)
akVs2PFJets.doAreaFastjet = cms.bool(True)
akVs3PFJets.doAreaFastjet = cms.bool(True)
akVs4PFJets.doAreaFastjet = cms.bool(True)
akVs5PFJets.doAreaFastjet = cms.bool(True)
akVs6PFJets.doAreaFastjet = cms.bool(True)

ak1PFJets = ak5PFJets.clone(rParam       = cms.double(0.1))
ak2PFJets = ak5PFJets.clone(rParam       = cms.double(0.2))
ak3PFJets = ak5PFJets.clone(rParam       = cms.double(0.3))
ak4PFJets = ak5PFJets.clone(rParam       = cms.double(0.4))
ak6PFJets = ak5PFJets.clone(rParam       = cms.double(0.6))

akVs1PFJets.jetPtMin = 1
akVs1PFJets.src = cms.InputTag("particleFlowTmp")
akVs1CaloJets.jetPtMin = 1
akVs2PFJets.jetPtMin = 1
akVs2PFJets.src = cms.InputTag("particleFlowTmp")
akVs2CaloJets.jetPtMin = 1
akVs3PFJets.jetPtMin = 1
akVs3PFJets.src = cms.InputTag("particleFlowTmp")
akVs3CaloJets.jetPtMin = 1
akVs4PFJets.jetPtMin = 1
akVs4PFJets.src = cms.InputTag("particleFlowTmp")
akVs4CaloJets.jetPtMin = 1
akVs5PFJets.jetPtMin = 1
akVs5PFJets.src = cms.InputTag("particleFlowTmp")
akVs5CaloJets.jetPtMin = 1
akVs6PFJets.jetPtMin = 1
akVs6PFJets.src = cms.InputTag("particleFlowTmp")
akVs6CaloJets.jetPtMin = 1
akPu1PFJets.jetPtMin = 1
akPu1CaloJets.jetPtMin = 1
akPu2PFJets.jetPtMin = 1
akPu2CaloJets.jetPtMin = 1
akPu3PFJets.jetPtMin = 1
akPu3CaloJets.jetPtMin = 1
akPu4PFJets.jetPtMin = 1
akPu4CaloJets.jetPtMin = 1
akPu5PFJets.jetPtMin = 1
akPu5CaloJets.jetPtMin = 1
akPu6PFJets.jetPtMin = 1
akPu6CaloJets.jetPtMin = 1
ak1PFJets.jetPtMin = 1
ak1PFJets.src = cms.InputTag("particleFlowTmp")
ak1CaloJets.jetPtMin = 1
ak2PFJets.jetPtMin = 1
ak2PFJets.src = cms.InputTag("particleFlowTmp")
ak2CaloJets.jetPtMin = 1
ak3PFJets.jetPtMin = 1
ak3PFJets.src = cms.InputTag("particleFlowTmp")
ak3CaloJets.jetPtMin = 1
ak4PFJets.jetPtMin = 1
ak4PFJets.src = cms.InputTag("particleFlowTmp")
ak4CaloJets.jetPtMin = 1
ak5PFJets.jetPtMin = 1
ak5PFJets.src = cms.InputTag("particleFlowTmp")
ak5CaloJets.jetPtMin = 1
ak6PFJets.jetPtMin = 1
ak6PFJets.src = cms.InputTag("particleFlowTmp")

hiReRecoPFJets = cms.Sequence(
akPu1PFJets
+
akPu2PFJets
+
akPu3PFJets
+
akPu4PFJets
+
akPu5PFJets
+
akPu6PFJets
+
akVs1PFJets
+
akVs2PFJets
+
akVs3PFJets
+
akVs4PFJets
+
akVs5PFJets
+
akVs6PFJets
+
ak1PFJets
+
ak2PFJets
+
ak3PFJets
+
ak4PFJets
+
ak5PFJets
+
ak6PFJets
+
akCs1PFJets
+
akCs2PFJets
+
akCs3PFJets
+
akCs4PFJets
+
akCs5PFJets
+
akCs6PFJets
+
akCsSoftDrop4PFJets
+
akCsSoftDrop5PFJets
+
akCsSoftDrop6PFJets
+
akCsFilter4PFJets
+
akCsFilter5PFJets
+
akCsFilter6PFJets
)

hiReRecoCaloJets = cms.Sequence(
akPu1CaloJets
+
akPu2CaloJets
+
akPu3CaloJets
+
akPu4CaloJets
+
akPu5CaloJets
+
akPu6CaloJets
+
akVs1CaloJets
+
akVs2CaloJets
+
akVs3CaloJets
+
akVs4CaloJets
+
akVs5CaloJets
+
akVs6CaloJets
+
ak1CaloJets
+
ak2CaloJets
+
ak3CaloJets
+
ak4CaloJets
+
ak5CaloJets
+
ak6CaloJets
)
