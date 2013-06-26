import FWCore.ParameterSet.Config as cms

# File: genMetCalo.cff
# Author: R. Remington
#
# Form Missing ET from Generator Information and store into event as a GenMET
# product.  Exclude calo invisible final state particles like neutrinos, muons
genMetCalo = cms.EDProducer("METProducer",
                            src = cms.InputTag("genCandidatesForMET"), ## Input  product label		  
                            
                            METType = cms.string('GenMET'), ## Output MET type		  
                            
                            alias = cms.string('GenMETCalo'), ## Alias  for FWLite		  
                            
                            onlyFiducialParticles = cms.bool(True), ## Use Only Fiducial Gen Particles
                            
                            globalThreshold = cms.double(0.0), ## Global Threshold for input objects
                            
                            usePt   = cms.bool(True), ## using Pt instead Et

                            applyFiducialThresholdForFractions   = cms.bool(False),

                            InputType = cms.string('CandidateCollection') ## Input  product type		  

)


