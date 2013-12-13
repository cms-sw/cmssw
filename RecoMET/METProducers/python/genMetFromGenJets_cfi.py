import FWCore.ParameterSet.Config as cms

# File: genMetFromGenJets.cfi
# Author: B. Scurlock
# Date: 03.01.2008
#
# Form Missing ET from Generator Information and store into event as a GenMET
# product.  
genMetAK5GenJets = cms.EDProducer("METProducer",
    src = cms.InputTag("ak4GenJets"), ## Input  product label		  

    METType = cms.string('MET'), ## Output MET type		  

    alias = cms.string('GenMETAK5'), ## Alias  for FWLite		  

    noHF = cms.bool(False), ## do not exclude HF

    globalThreshold = cms.double(0.0), ## Global Threshold for input objects

    usePt   = cms.bool(True), ## using Pt instead Et

    applyFiducialThresholdForFractions   = cms.bool(False),

    InputType = cms.string('CandidateCollection') ## Input  product type		  

)


