import FWCore.ParameterSet.Config as cms

# File: genMetCaloAndNonPrompt.cff
# Author: R. Remington
#
# Form Missing ET from Generator Information and store into event as a GenMET
# product.  Exclude calo invisible, non-resonant, final state particles like neutrinos, muons
genMetCaloAndNonPrompt = cms.EDProducer("METProducer",
    src = cms.InputTag("genParticlesForJets"), ## Input  product label		  

    METType = cms.string('GenMET'), ## Output MET type		  

    alias = cms.string('GenMETCaloAndNonPrompt'), ## Alias  for FWLite		  

    onlyFiducialParticles = cms.bool(True), ## use only fiducial GenParticles

    globalThreshold = cms.double(0.0), ## Global Threshold for input objects

    usePt   = cms.bool(True), ## using Pt instead Et

    applyFiducialThresholdForFractions   = cms.bool(False),

    InputType = cms.string('CandidateCollection') ## Input  product type		  

)


