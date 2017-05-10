#!/usr/bin/env python
"""
_trackingRun2noCCC_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)

class trackingRun2noCCC(pp):
    def __init__(self):
        
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
    """
    _trackingRun2noCCC_

    Implement configuration building for data processing for proton
    collision data taking without CCC in the tracking sequence

    """
    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
#        if not 'skims' in args:
#            args['skims']=['SiStripAlCaMinBias']

        if not 'customs' in args:
            args['customs']=['Configuration/DataProcessing/RecoTLR.customisePromptRun2']
        else:
            args['customs'].append('Configuration/DataProcessing/RecoTLR.customisePromptRun2')

        process = pp.promptReco(self,globalTag,**args)

#        for pset in process._Process__psets.values():
#            if hasattr(pset,'ComponentType'):
#                if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                    if hasattr(pset,'minGoodStripCharge'):
#                        pset.minGoodStripCharge = cms.PSet( refToPSet_ = cms.string('SiStripClusterChargeCutNone') )
#                if (pset.ComponentType == 'MaxCCCLostHitsTrajectoryFilter'):
#                    if hasattr(pset,'minGoodStripCharge'):
#                        pset.minGoodStripCharge = cms.PSet( refToPSet_ = cms.string('SiStripClusterChargeCutNone') )
#    
#        for module in esproducers_by_type(process,'Chi2ChargeMeasurementEstimatorESProducer'):
#            module.clusterChargeCut = cms.PSet( refToPSet_ = cms.string('SiStripClusterChargeCutNone') )

        # overwrite parameter which handles the CCC value            
        if hasattr(process,'SiStripClusterChargeCutTiny'):
            setattr(process,'SiStripClusterChargeCutTiny', cms.PSet(value = cms.double( -1.0 ) ) )  #  800.0
        if hasattr(process,'SiStripClusterChargeCutLoose'):
            setattr(process,'SiStripClusterChargeCutLoose', cms.PSet(value = cms.double( -1.0 ) ) ) # 1620.0
        if hasattr(process,'SiStripClusterChargeCutTight'):
            setattr(process,'SiStripClusterChargeCutTight', cms.PSet(value = cms.double( -1.0 ) ) ) # 1945.0
        
        return process
