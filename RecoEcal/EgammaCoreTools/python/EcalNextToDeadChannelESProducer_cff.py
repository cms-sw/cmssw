#
# Configure the EcalNextToDeadChannel service 
#
# Author: Stefano Argiro
#


import FWCore.ParameterSet.Config as cms

essourceEcalNextToDead =  cms.ESSource("EmptyESSource",
                           recordName = cms.string("EcalNextToDeadChannelRcd"),
                           firstValid = cms.vuint32(1),
                           iovIsRunNotTime = cms.bool(True)
                                       )


ecalNextToDeadChannelESProducer = cms.ESProducer("EcalNextToDeadChannelESProducer",
                            # channels with db status >= this threshold will
                            # be defined as dead                         
                            channelStatusThresholdForDead=cms.int32(12)
                                                 )
# foo bar baz
# LD1OcYB03IcQd
# Mv7A8VsS0Buem
