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
 # for some reason hex notation does not seem to work with vuint32
                                       
 flagMask=cms.vuint32( 1,    #0x00000001, # good->good
                       34,   #0x00000022, # poorreco,poorcalib->problematic
                       896,  #0x00000380, # LERecovered,TowRecovered->recovered
                       4,    #0x00000004, # outoftime->time 
                       49152,#0x0000C000, # weird,diweird->weird
                       6232  #0x00001858  # faultyhw,noisy,saturated,dead,killed->bad
 ),
 # map ChannelStatus flags into EcalSeverityLevel
 dbstatusMask=cms.vuint32( 1,   #0x00000001, # good-> good;
                           2046,#0x000007FE, # status 1..10 -> problematic
                           0,   #0x00000000, # nothing->recovered
                           0,   #0x00000000, # nothing->time
                           0,   #0x00000000, #nothing->weird
                           64512#0x0000FC00  #status 11..16 ->bad
     ),
 #return kTime only if the rechit is above this threshold            
 timeThresh=cms.double(2.0)                                  
 )
