# THIS CONFIGURATION IS BROKEN. SINCE 2015 Geometry_cff has been deleted
# and it is a fatal error to load it. And because of this I did not bother
# to convert it to use tasks in 2017 when Tasks where implemented for unscheduled
# mode (or remove the allowUnscheduled flag which no longer does anything).
# Modules which are supposed to run unscheduled will not run.  Someone should
# probably either fix or delete this ...

import FWCore.ParameterSet.Config as cms

process = cms.Process("S2")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:patTuple_mini.root")
)
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarPileUpMINIAODSIM
process.source.fileNames = filesRelValProdTTbarPileUpMINIAODSIM

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
from RecoMET.METProducers.pfMet_cfi import pfMet

#select isolated collections
process.selectedMuons = cms.EDFilter("CandPtrSelector", src = cms.InputTag("slimmedMuons"), cut = cms.string('''abs(eta)<2.5 && pt>10. &&
   (pfIsolationR04().sumChargedHadronPt+
    max(0.,pfIsolationR04().sumNeutralHadronEt+
    pfIsolationR04().sumPhotonEt-
    0.50*pfIsolationR04().sumPUPt))/pt < 0.20 && 
    (isPFMuon && (isGlobalMuon || isTrackerMuon) )'''))
process.selectedElectrons = cms.EDFilter("CandPtrSelector", src = cms.InputTag("slimmedElectrons"), cut = cms.string('''abs(eta)<2.5 && pt>20. &&
    gsfTrack.isAvailable() &&
    gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') < 2 &&
    (pfIsolationVariables().sumChargedHadronPt+
    max(0.,pfIsolationVariables().sumNeutralHadronEt+
    pfIsolationVariables().sumPhotonEt-
    0.5*pfIsolationVariables().sumPUPt))/pt < 0.15'''))

#do projections
process.load("CommonTools.ParticleFlow.pfCHS_cff")
process.pfNoMuonCHS =  cms.EDProducer("CandPtrProjector", src = cms.InputTag("pfCHS"), veto = cms.InputTag("selectedMuons"))
process.pfNoElectronsCHS = cms.EDProducer("CandPtrProjector", src = cms.InputTag("pfNoMuonCHS"), veto =  cms.InputTag("selectedElectrons"))

process.pfNoMuon =  cms.EDProducer("CandPtrProjector", src = cms.InputTag("packedPFCandidates"), veto = cms.InputTag("selectedMuons"))
process.pfNoElectrons = cms.EDProducer("CandPtrProjector", src = cms.InputTag("pfNoMuon"), veto =  cms.InputTag("selectedElectrons"))



process.ak4PFJets = ak4PFJets.clone(src = 'pfNoElectrons', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak4PFJetsCHS = ak4PFJets.clone(src = 'pfNoElectronsCHS', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak4GenJets = ak4GenJets.clone(src = 'packedGenParticles')


# The following is make patJets, but EI is done with the above
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic')


from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
addJetCollection(
   process,
   postfix   = "",
   labelName = 'AK4PFCHS',
   jetSource = cms.InputTag('ak4PFJetsCHS'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   pfCandidates = cms.InputTag('packedPFCandidates'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   btagDiscriminators = [ 'pfCombinedSecondaryVertexBJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags' ],
   genJetCollection=cms.InputTag('ak4GenJets'),
   genParticles=cms.InputTag('prunedGenParticles')
   )
addJetCollection(
   process,
   postfix   = "",
   labelName = 'AK4PF',
   jetSource = cms.InputTag('ak4PFJets'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   pfCandidates = cms.InputTag('packedPFCandidates'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   jetCorrections = ('AK4PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   btagDiscriminators = [ 'pfCombinedSecondaryVertexBJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags' ],
   genJetCollection=cms.InputTag('ak4GenJets'),
   genParticles=cms.InputTag('prunedGenParticles')
   )

# if using legacy jet flavour (not used by default)
process.patJetPartonsLegacy.skipFirstN = cms.uint32(0) # do not skip first 6 particles, we already pruned some!
process.patJetPartonsLegacy.acceptNoDaughters = cms.bool(True) # as we drop intermediate stuff, we need to accept quarks with no siblings


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.suppressWarning = cms.untracked.vstring('ecalLaserCorrFilter','manystripclus53X','toomanystripclus53X')
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.options.allowUnscheduled = cms.untracked.bool(True)

process.OUT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(['drop *','keep patJets_patJetsAK4PF_*_*','keep patJets_patJetsAK4PFCHS_*_*','keep *_*_*_PAT'])
)
process.endpath= cms.EndPath(process.OUT)

