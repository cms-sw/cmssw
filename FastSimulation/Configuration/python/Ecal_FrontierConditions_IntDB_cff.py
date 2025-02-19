# The following comments couldn't be translated into the new config version:

# Integration database

import FWCore.ParameterSet.Config as cms

#
# Ecal  calibrations from Frontier
#
from RecoLocalCalo.EcalRecProducers.getEcalConditions_frontier_cff import *
ecalConditions.connect = 'frontier://cms_conditions_data/CMS_COND_20X_ECAL'

