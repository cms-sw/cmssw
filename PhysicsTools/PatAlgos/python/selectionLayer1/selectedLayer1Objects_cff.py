import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import *

#from PhysicsTools.PatAlgos.producersLayer1.hemisphereProducer_cfi import *

# One module to count objects
selectedLayer1Summary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("selectedLayer1Objects|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("selectedLayer1Electrons"),
        cms.InputTag("selectedLayer1Muons"),
        cms.InputTag("selectedLayer1Taus"),
        cms.InputTag("selectedLayer1Photons"),
        cms.InputTag("selectedLayer1Jets"),
    )
)


selectedLayer1Objects = cms.Sequence(
    selectedLayer1Electrons +
    selectedLayer1Muons +
    selectedLayer1Taus +
    selectedLayer1Photons +
    selectedLayer1Jets +
 #  selectedLayer1Hemispheres +
    selectedLayer1Summary
)
