import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters


from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

##############################################################################
# Standard AK8 Jets####################################################
##########################
ak8PFJets = ak4PFJets.clone( 
    rParam   = 0.8,
    jetPtMin = 50.0 
    )

##############################################################################
# AK8 jets with various pileup subtraction schemes
##############################################################################
ak8PFJetsPuppi = ak8PFJets.clone(
    src = "particleFlow",
    applyWeight = True,
    srcWeights  = cms.InputTag("puppi")
    )

ak8PFJetsCHS = ak8PFJets.clone(
    src = "pfNoPileUpJME"
    )

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(ak8PFJetsCHS, src = "pfEmptyCollection")
pp_on_AA.toModify(ak8PFJetsPuppi, src = "pfEmptyCollection")

ak8PFJetsCS = ak8PFJets.clone(
    useConstituentSubtraction = cms.bool(True),    
    csRParam = cms.double(0.4),
    csRho_EtaMax = ak8PFJets.Rho_EtaMax,   # Just use the same eta for both C.S. and rho by default
    useExplicitGhosts = cms.bool(True),
    doAreaFastjet = True,
    jetPtMin = 100.0
    )


##############################################################################
# Preclustered constituents for substructure, various subtraction schemes
##############################################################################
ak8PFJetsCSConstituents = cms.EDProducer("PFJetConstituentSelector",
                                         src = cms.InputTag("ak8PFJetsCS"),
                                         cut = cms.string("pt > 100.0")
                                        )

ak8PFJetsCHSConstituents = cms.EDProducer("PFJetConstituentSelector",
                                          src = cms.InputTag("ak8PFJetsCHS"),
                                          cut = cms.string("pt > 100.0 && abs(rapidity()) < 2.4")
                                         )

ak8PFJetsPuppiConstituents = cms.EDProducer("PFJetConstituentSelector",
                                          src = cms.InputTag("ak8PFJetsPuppi"),
                                          cut = cms.string("pt > 100.0 && abs(rapidity()) < 2.4")
                                         )


##############################################################################
# Substructure algorithms
##############################################################################
ak8PFJetsCHSFiltered = ak8PFJets.clone(
    src = "ak8PFJetsCHSConstituents:constituents",
    useFiltering = cms.bool(True),
    nFilt = cms.int32(3),
    rFilt = cms.double(0.3),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets"),
    jetPtMin = 100.0
    )


ak8PFJetsCHSMassDropFiltered = ak8PFJets.clone(
    src = "ak8PFJetsCHSConstituents:constituents",
    useMassDropTagger = cms.bool(True),
    muCut = cms.double(0.667),
    yCut = cms.double(0.08),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets"),
    jetPtMin = 100.0
    )

ak8PFJetsCHSPruned = ak8PFJets.clone(
    SubJetParameters,
    src = "ak8PFJetsCHSConstituents:constituents",
    usePruning = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets"),
    jetPtMin = 100.0,
    doAreaFastjet = False
    )

ak8PFJetsCHSSoftDrop = ak8PFJets.clone(
    useSoftDrop = cms.bool(True),
    src = "ak8PFJetsCHSConstituents:constituents",
    zcut = cms.double(0.1),
    beta = cms.double(0.0),
    R0   = cms.double(0.8),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets"),
    jetPtMin = 100.0
    )


ak8PFJetsCHSTrimmed = ak8PFJets.clone(
    useTrimming = cms.bool(True),
    src = "ak8PFJetsCHSConstituents:constituents",
    rFilt = cms.double(0.2),
    trimPtFracMin = cms.double(0.03),
    useExplicitGhosts = cms.bool(True),
    jetPtMin = 100.0
    )

ak8PFJetsPuppiSoftDrop = ak8PFJetsCHSSoftDrop.clone(
    src = "ak8PFJetsPuppiConstituents:constituents",
    applyWeight = True,
    srcWeights = cms.InputTag("puppi")
    )
