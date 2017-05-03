import FWCore.ParameterSet.Config as cms

### PP RECO does not include R=3 or R=5 jets.
### re-RECO is only possible for PF, RECO is missing calotowers
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
ak5PFJets.doAreaFastjet = True
ak3PFJets = ak5PFJets.clone(rParam = 0.3)
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
ak3GenJets = ak5GenJets.clone(rParam = 0.3)

from RecoJets.Configuration.GenJetParticles_cff import *
from RecoHI.HiJetAlgos.HiGenJets_cff import *
from HeavyIonsAnalysis.JetAnalysis.makePartons_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_pp_mc_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pp_mc_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_pp_mc_cff import *
from HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pp_mc_cff import *

highPurityTracks = cms.EDFilter("TrackSelector",
                                src = cms.InputTag("generalTracks"),
                                cut = cms.string('quality("highPurity")')
)

# Other radii jets and calo jets need to be reconstructed
jetSequences = cms.Sequence(
    genParticlesForJets +
    ak3GenJets +
    ak5GenJets +
    ak3PFJets +
    ak5PFJets +
    makePartons +
    highPurityTracks +
    ak3PFJetSequence +
    ak4PFJetSequence +
    ak5PFJetSequence +
    ak4CaloJetSequence
)
