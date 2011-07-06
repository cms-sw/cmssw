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
    kGood       = cms.vuint32(0),
    kProblematic= cms.vuint32(1,2,3,4,5,6,7,8,9,10),
    kRecovered  = cms.vuint32(),
    kTime       = cms.vuint32(),
    kWeird      = cms.vuint32(),
    kBad        = cms.vuint32(11,12,13,14,15,16)
     ),

 #return kTime only if the rechit is above this threshold            
 timeThresh=cms.double(2.0),

  
                                   
 )
