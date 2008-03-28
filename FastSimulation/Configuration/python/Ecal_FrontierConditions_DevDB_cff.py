# The following comments couldn't be translated into the new config version:

# For development data base

import FWCore.ParameterSet.Config as cms

#
# Ecal  calibrations from Frontier
#
from RecoLocalCalo.EcalRecProducers.getEcalConditions_frontier_cff import *
ecalConditions.connect = 'frontier://FrontierDev/CMS_COND_ECAL'

