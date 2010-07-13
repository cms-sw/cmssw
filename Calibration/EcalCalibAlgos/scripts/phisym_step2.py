#
# Python config file to drive phi symmetry et sum accumulation
#
# Author: Stefano Argiro
#

import FWCore.ParameterSet.Config as cms

process=cms.Process("PHISYMCAL")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#an input file is with the same event iovs as for the calibration is
#needed
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_GOODCOLL-v1/0140/E0F8D4D2-C83E-DF11-9249-002618943877.root',
      '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_GOODCOLL-v1/0140/F4EA88D2-C83E-DF11-AF55-00261894392C.root'

    )
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# Global Tag
process.GlobalTag.globaltag = 'MC_3XY_V26::All'


process.phisymcalib = cms.EDAnalyzer("PhiSymmetryCalibration_step2",
                                      
    #channel statuses to be excluded                                  
    statusThreshold = cms.untracked.int32(0),
    #do we have an MC miscalibration to calculate expected precision ?    
    haveInitialMiscalib  = cms.untracked.bool(False),                     
    #name of the initial micalibration files
    initialmiscalibfile  = cms.untracked.string("InitialMiscalib.xml"),
    #are we reiterating ?
    reiteration          = cms.untracked.bool(False),     
    #when reiterating, old calib file                                 
    oldcalibfile    = cms.untracked.string("EcalIntercalibConstants.xml"), 

  )

process.p = cms.Path(process.phisymcalib)
