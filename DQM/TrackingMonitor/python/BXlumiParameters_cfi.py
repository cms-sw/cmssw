import FWCore.ParameterSet.Config as cms

# BXlumi                          
BXlumiSetup = cms.PSet(
    # input tags
    lumi    = cms.InputTag('lumiProducer'),
  # taken from 
  # DPGAnalysis/SiStripTools/src/DigiLumiCorrHistogramMaker.cc
  # the scale factor 6.37 should follow the lumi prescriptions
  # AS SOON AS THE CORRECTED LUMI WILL BE AVAILABLE IT HAS TO BE SET TO 1.
    lumiScale = cms.double(6.37),    
# low PU
#    BXlumiBin = cms.int32 (100),
#    BXlumiMin = cms.double(1),  
#    BXlumiMax = cms.double(10), 

    BXlumiBin = cms.int32 (400),
    BXlumiMin = cms.double(2000),
    BXlumiMax = cms.double(6000)    
)

