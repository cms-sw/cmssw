import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_HI_cff import *
from RecoHI.HiJetAlgos.HiGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from Configuration.StandardSequences.ReconstructionHeavyIons_cff import voronoiBackgroundPF, voronoiBackgroundCalo

akHiGenJets = cms.Sequence(
    genParticlesForJets +
    ak1HiGenJets +
    ak2HiGenJets +
    ak3HiGenJets +
    ak4HiGenJets +
    ak5HiGenJets +
    ak6HiGenJets)

from HeavyIonsAnalysis.JetAnalysis.jets.akPu1CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs1CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak1CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs1PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu1PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak1PFJetSequence_PbPb_jec_cff import *

from HeavyIonsAnalysis.JetAnalysis.jets.akPu2CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs2CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak2CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs2PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu2PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak2PFJetSequence_PbPb_jec_cff import *

from HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_PbPb_jec_cff import *

from HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_PbPb_jec_cff import *

from HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_PbPb_jec_cff import *

from HeavyIonsAnalysis.JetAnalysis.jets.akPu6CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs6CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak6CaloJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akVs6PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.akPu6PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak6PFJetSequence_PbPb_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.makePartons_cff import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
offlinePrimaryVertices.TrackLabel = 'highPurityTracks'

highPurityTracks = cms.EDFilter("TrackSelector",
                                src = cms.InputTag("hiGeneralTracks"),
                                cut = cms.string('quality("highPurity")')
)

akVs1PFJetAnalyzer.doSubEvent = True
akVs1CaloJetAnalyzer.doSubEvent = True

akVs2PFJetAnalyzer.doSubEvent = True
akVs2CaloJetAnalyzer.doSubEvent = True

akVs3PFJetAnalyzer.doSubEvent = True
akVs3CaloJetAnalyzer.doSubEvent = True

akVs4PFJetAnalyzer.doSubEvent = True
akVs4CaloJetAnalyzer.doSubEvent = True

akVs5PFJetAnalyzer.doSubEvent = True
akVs5CaloJetAnalyzer.doSubEvent = True

akVs6PFJetAnalyzer.doSubEvent = True
akVs6CaloJetAnalyzer.doSubEvent = True

akPu1PFJetAnalyzer.doSubEvent = True
akPu1CaloJetAnalyzer.doSubEvent = True

akPu2PFJetAnalyzer.doSubEvent = True
akPu2CaloJetAnalyzer.doSubEvent = True

akPu3PFJetAnalyzer.doSubEvent = True
akPu3CaloJetAnalyzer.doSubEvent = True

akPu4PFJetAnalyzer.doSubEvent = True
akPu4CaloJetAnalyzer.doSubEvent = True

akPu5PFJetAnalyzer.doSubEvent = True
akPu5CaloJetAnalyzer.doSubEvent = True

akPu6PFJetAnalyzer.doSubEvent = True
akPu6CaloJetAnalyzer.doSubEvent = True

ak1PFJetAnalyzer.doSubEvent = True
ak1CaloJetAnalyzer.doSubEvent = True

ak2PFJetAnalyzer.doSubEvent = True
ak2CaloJetAnalyzer.doSubEvent = True

ak3PFJetAnalyzer.doSubEvent = True
ak3CaloJetAnalyzer.doSubEvent = True

ak4PFJetAnalyzer.doSubEvent = True
ak4CaloJetAnalyzer.doSubEvent = True

ak5PFJetAnalyzer.doSubEvent = True
ak5CaloJetAnalyzer.doSubEvent = True

ak6PFJetAnalyzer.doSubEvent = True
ak6CaloJetAnalyzer.doSubEvent = True



jetSequences = cms.Sequence(akHiGenJets +
                            voronoiBackgroundPF+
                            voronoiBackgroundCalo+

                            hiReRecoCaloJets +
                            hiReRecoPFJets +
                            
							myPartons +
							highPurityTracks +
							offlinePrimaryVertices +
							
                    	    ak1CaloJetSequence +
                            akPu1CaloJetSequence +
                            akVs1CaloJetSequence +
                            ak1PFJetSequence +
                            akVs1PFJetSequence +
                            akPu1PFJetSequence +

			                ak2CaloJetSequence +
			                akPu2CaloJetSequence +
                            akVs2CaloJetSequence +
                            akVs2PFJetSequence +
                            akPu2PFJetSequence +
                            ak2PFJetSequence +

			                ak3CaloJetSequence +
			                akPu3CaloJetSequence +
                            akVs3CaloJetSequence +
                            ak3PFJetSequence +
                            akVs3PFJetSequence +
                            akPu3PFJetSequence +

                            ak4CaloJetSequence +
                            akPu4CaloJetSequence +
                            akVs4CaloJetSequence +
                            ak4PFJetSequence +
                            akVs4PFJetSequence +
                            akPu4PFJetSequence +

                            ak5CaloJetSequence +
                            akPu5CaloJetSequence +
                            akVs5CaloJetSequence +
                            ak5PFJetSequence +
                            akVs5PFJetSequence +
                            akPu5PFJetSequence +

                            ak6CaloJetSequence +
                            akPu6CaloJetSequence +
                            akVs6CaloJetSequence +
                            ak6PFJetSequence +
                            akVs6PFJetSequence +
                            akPu6PFJetSequence
)
