import FWCore.ParameterSet.Config as cms

# File: genMetTrue_cfi.py
# Author: R. Remington
#
# Form Missing ET from Generator Information and store into event as a GenMET
# product.  Exclude only invisible final state particles like neutrinos. 
genMetTrue = cms.EDProducer("METProducer",
                            src = cms.InputTag("genParticlesForMETAllVisible"), ## Input  product label		  
                            
                            METType = cms.string('GenMET'), ## Output MET type		  
                            
                            alias = cms.string('GenMETAllVisible'), ## Alias  for FWLite		  
                            
                            onlyFiducialParticles = cms.bool(False), ## Use only fiducial GenParticles
                            
                            globalThreshold = cms.double(0.0), ## Global Threshold for input objects
                            
                            usePt   = cms.bool(True), ## using Pt instead Et

                            applyFiducialThresholdForFractions   = cms.bool(False),
                            
                            InputType = cms.string('CandidateCollection') ## Input  product type		  
                            
                            )


