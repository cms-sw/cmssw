#
# Configure the EcalSeverityLevel service 
#
# Author: Stefano Argiro
#


import FWCore.ParameterSet.Config as cms

EcalSeverityLevelService = cms.Service("EcalSeverityLevelService",

 # map EcalRecHit::Flag into EcalSeverityLevel
 # for some reason hex notation does not seem to work with vuint32
                                       
 flagMask=cms.vuint32( 1,    #0x00000001, # good->good
                       34,   #0x00000022, # poorreco,poorcalib->problematic
                       384,  #0x00000180, # LERecovered,TowRecovered->recovered
                       4,    #0x00000004, # outoftime->time 
                       24576,#0x00006000, # weird,diweird->weird
                       1688  #0x00000698  # faultyhw,noisy,saturated,dead,killed->bad
 ),
 # map ChannelStatus flags into EcalSeverityLevel
 dbstatusMask=cms.vuint32( 1,   #0x00000001, # good-> good;
                           2046,#0x000007FE, # status 1..10 -> problematic
                           0,   #0x00000000, # nothing->recovered
                           0,   #0x00000000, # nothing->time
                           0,   #0x00000000, #nothing->weird
                           64512#0x0000FC00  #status 11..16 ->bad
     )                                   
 )
