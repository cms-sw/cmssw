import FWCore.ParameterSet.Config as cms

# CMSSW version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

if cmsswVersion >= "CMSSW_7_5":

# Simplified TrackerTopologyEP config (PR #7966)
    if 'trackerTopology' in locals():
        paras = tuple(vars(trackerTopology)['_Parameterizable__parameterNames'])
        for para in paras:
          delattr(trackerTopology,para)
        setattr(trackerTopology,'appendToDataLabel',cms.string(""))

# Removal of 'upgradeGeometry' from TrackerDigiGeometryESModule (PR #7794)
    if 'TrackerDigiGeometryESModule' in locals():
        if  hasattr(TrackerDigiGeometryESModule.trackerGeometryConstants,'upgradeGeometry'):
            delattr(TrackerDigiGeometryESModule.trackerGeometryConstants,'upgradeGeometry')
