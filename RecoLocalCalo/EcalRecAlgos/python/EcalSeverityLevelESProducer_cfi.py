#
# Configure the EcalSeverityLevel service 
#
# Author: Stefano Argiro
#


import FWCore.ParameterSet.Config as cms

essourceEcalSev =  cms.ESSource("EmptyESSource",
                    recordName = cms.string("EcalSeverityLevelAlgoRcd"),
                    firstValid = cms.vuint32(1),
                    iovIsRunNotTime = cms.bool(True)
                    )


ecalSeverityLevel = cms.ESProducer("EcalSeverityLevelESProducer",

 # map EcalRecHit::Flag into EcalSeverityLevel
 flagMask = cms.PSet (
    kGood       = cms.vstring('kGood'),
    kProblematic= cms.vstring('kPoorReco','kPoorCalib','kNoisy','kSaturated'),
    kRecovered  = cms.vstring('kLeadingEdgeRecovered','kTowerRecovered'),
    kTime       = cms.vstring('kOutOfTime'),
    kWeird      = cms.vstring('kWeird','kDiWeird'),
    kBad        = cms.vstring('kFaultyHardware','kDead','kKilled')
     ),                                                                   
 
 # map ChannelStatus flags into EcalSeverityLevel
 dbstatusMask=cms.PSet(
    kGood       = cms.vstring('kOk'),
    kProblematic= cms.vstring('kDAC',
                              'kNoLaser',
                              'kNoisy',
                              'kNNoisy',
                              'kNNNoisy',
                              'kNNNNoisy',
                              'kNNNNNoisy',
                              'kFixedG6',
                              'kFixedG1',
                              'kFixedG0',),
    kRecovered  = cms.vstring(),
    kTime       = cms.vstring(),
    kWeird      = cms.vstring(),
    kBad        = cms.vstring('kNonRespondingIsolated',
                              'kDeadVFE',
                              'kDeadFE',
                              'kNoDataNoTP')
     ),

 #return kTime only if the rechit is above this threshold            
 timeThresh=cms.double(2.0),

  
                                   
 )
