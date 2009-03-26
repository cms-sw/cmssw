import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
from PhysicsTools.PatAlgos.recoLayer0.electronId_cff import *
from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *
from PhysicsTools.PatAlgos.recoLayer0.muonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.photonId_cff import *
from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauDiscriminators_cff import *

# These two duplicate removals are here because they're AOD bugfixes
from PhysicsTools.PatAlgos.recoLayer0.duplicatedElectrons_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.duplicatedPhotons_cfi   import *

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *  # needed for the MET

# Sequences needed to deliver the objects
# You shouldn't remove modules from here unless you *know* what you're doing
patAODCoreReco = cms.Sequence(
    electronsNoDuplicates 
)

# Sequences needed to deliver external information for objects
# You can remove modules from here if you don't need these features
patAODExtraReco = cms.Sequence(
    #patBTagging +       # Empty sequences not supported yet
    patElectronId +
    patElectronIsolation +
    patJetMETCorrections +
    patJetTracksCharge +
    #patMuonIsolation +   # Empty sequences not supported yet
    #patPhotonID +        # Empty sequences not supported yet
    patPhotonIsolation 
    #patTauDiscrimination # Empty sequences not supported yet
)

# One module to count some AOD Objects that are usually input to PAT
aodSummary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("aodObjects|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("pixelMatchGsfElectrons"),
        cms.InputTag("electronsNoDuplicates"),
        cms.InputTag("muons"),
        cms.InputTag("caloRecoTauProducer"),
        cms.InputTag("pfRecoTauProducer"),
        cms.InputTag("photons"),
        cms.InputTag("iterativeCone5CaloJets"),
        cms.InputTag("met"),
    )
)
#aodContents = cms.EDAnalyzer("EventContentAnalyzer")

# Default PAT reconstruction sequence on top of AOD
patAODReco = cms.Sequence(
    patAODCoreReco +
    patAODExtraReco +
    aodSummary #+aodContents
)
