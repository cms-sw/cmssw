import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.jets.ak1PFJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak1CaloJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak2PFJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak2CaloJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak6PFJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak6CaloJetSequence_pp_jec_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_pp_cff import *
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
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
ak5GenJets = ak5GenJets
ak1GenJets = ak5GenJets.clone(rParam = 0.1)
ak2GenJets = ak5GenJets.clone(rParam = 0.2)
ak3GenJets = ak5GenJets.clone(rParam = 0.3)
ak4GenJets = ak5GenJets.clone(rParam = 0.4)
ak6GenJets = ak5GenJets.clone(rParam = 0.6)
from RecoJets.Configuration.GenJetParticles_cff import *

akGenJets = cms.Sequence(
    genParticlesForJets +
    ak1GenJets+
    ak2GenJets+
    ak3GenJets+
    ak4GenJets+
    ak5GenJets+
    ak6GenJets
)
from HeavyIonsAnalysis.JetAnalysis.makePartons_cff import *
highPurityTracks = cms.EDFilter("TrackSelector",
                                src = cms.InputTag("generalTracks"),
                                cut = cms.string('quality("highPurity")')
)

jetSequences = cms.Sequence(
    akGenJets +
    ppReRecoPFJets +
    ppReRecoCaloJets +
    makePartons +
    highPurityTracks +
    ak1PFJetSequence +
    ak1CaloJetSequence +
    ak2PFJetSequence +
    ak2CaloJetSequence +
    ak3PFJetSequence +
    ak3CaloJetSequence +
    ak4PFJetSequence +
    ak4CaloJetSequence +
    ak5PFJetSequence +
    ak5CaloJetSequence +
    ak6PFJetSequence +
    ak6CaloJetSequence)
