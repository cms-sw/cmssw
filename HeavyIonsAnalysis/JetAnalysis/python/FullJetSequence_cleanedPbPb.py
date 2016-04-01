import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_HI_cff import *
from Configuration.StandardSequences.ReconstructionHeavyIons_cff import voronoiBackgroundPF, voronoiBackgroundCalo
from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer
from RecoHI.HiJetAlgos.hiFJGridEmptyAreaCalculator_cff import hiFJGridEmptyAreaCalculator
kt4PFJets.src = cms.InputTag('particleFlowTmp')
kt4PFJets.doAreaFastjet = True
kt4PFJets.jetPtMin      = cms.double(0.0)
kt4PFJets.GhostArea     = cms.double(0.005)
kt2PFJets = kt4PFJets.clone(rParam       = cms.double(0.2))


from HeavyIonsAnalysis.JetAnalysis.jets.akPu2CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs2CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs2PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu2PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCs2PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCs3PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCs4PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCs5PFJetSequence_PbPb_mb_cff import *

from HeavyIonsAnalysis.JetAnalysis.jets.akCsFilter4PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCsFilter5PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCsFilter6PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCsSoftDrop4PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCsSoftDrop5PFJetSequence_PbPb_mb_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akCsSoftDrop6PFJetSequence_PbPb_mb_cff import *

highPurityTracks = cms.EDFilter("TrackSelector",
                                src = cms.InputTag("hiGeneralTracks"),
                                cut = cms.string('quality("highPurity")'))

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
offlinePrimaryVertices.TrackLabel = 'highPurityTracks'

jetSequences = cms.Sequence(
    voronoiBackgroundPF+
    voronoiBackgroundCalo+
    kt2PFJets +
    kt4PFJets +
    hiFJRhoProducer +
    hiFJGridEmptyAreaCalculator +

    akPu2CaloJets +
    akPu2PFJets +
    akVs2CaloJets +
    akVs2PFJets +
    akCs2PFJets +

    #akPu3CaloJets +
    #akPu3PFJets +
    akVs3CaloJets +
    akVs3PFJets +
    akCs3PFJets +

    #akPu4CaloJets +
    #akPu4PFJets +
    akVs4CaloJets +
    akVs4PFJets +
    akCs4PFJets +

    akPu5CaloJets +
    akPu5PFJets +
    akVs5CaloJets +
    akVs5PFJets +
    akCs5PFJets +

    akCsFilter4PFJets +
    akCsFilter5PFJets +
    akCsSoftDrop4PFJets +
    akCsSoftDrop5PFJets +

    highPurityTracks +
    offlinePrimaryVertices +

    akPu2CaloJetSequence +
    akVs2CaloJetSequence +
    akVs2PFJetSequence +
    akPu2PFJetSequence +
    akCs2PFJetSequence +

    akPu3CaloJetSequence +
    akVs3CaloJetSequence +
    akVs3PFJetSequence +
    akPu3PFJetSequence +
    akCs3PFJetSequence +

    akPu4CaloJetSequence +
    akVs4CaloJetSequence +
    akVs4PFJetSequence +
    akPu4PFJetSequence +
    akCs4PFJetSequence +

    akPu5CaloJetSequence +
    akVs5CaloJetSequence +
    akVs5PFJetSequence +
    akPu5PFJetSequence +
    akCs5PFJetSequence +

    akCsFilter4PFJetSequence +
    akCsFilter5PFJetSequence +
    akCsSoftDrop4PFJetSequence +
    akCsSoftDrop5PFJetSequence
)
