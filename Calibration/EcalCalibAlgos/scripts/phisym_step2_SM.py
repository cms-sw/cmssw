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
    fileNames = cms.untracked.vstring('/store/data/Commissioning10/AlCaPhiSymEcal/ALCARECO/v9/000/133/537/18F7774C-134C-DF11-980D-001D09F24489.root'

    )
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# Global Tag
process.GlobalTag.globaltag = 'GR_R_35X_V8A::All'


process.phisymcalib = cms.EDAnalyzer("PhiSymmetryCalibration_step2_SM",
                                      
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
