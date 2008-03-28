import FWCore.ParameterSet.Config as cms

# File: GenMET.cff
# Author: R. Cavanaugh
# Date: 08.08.2006
#
# Form Missing ET from Generator Information and store into event as a GenMET
# product.  Exclude calo invisible final state particles like neutrinos, muons
genMet = cms.EDProducer("METProducer",
    src = cms.InputTag("genCandidatesForMET"), ## Input  product label		  

    METType = cms.string('GenMET'), ## Output MET type		  

    alias = cms.string('GenMET'), ## Alias  for FWLite		  

    noHF = cms.bool(False), ## do not exclude HF	

    globalThreshold = cms.double(0.0), ## Global Threshold for input objects

    InputType = cms.string('CandidateCollection') ## Input  product type		  

)


