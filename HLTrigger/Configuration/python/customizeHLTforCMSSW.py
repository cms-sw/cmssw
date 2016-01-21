import FWCore.ParameterSet.Config as cms

#
# reusable functions
def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)

def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)

#
# one action function per PR - put the PR number into the name of the function

# Remove hcalTopologyConstants
def customiseFor11920(process):
    if hasattr(process,'HcalGeometryFromDBEP'):
        if hasattr(process.HcalGeometryFromDBEP,'hcalTopologyConstants'):
            delattr(process.HcalGeometryFromDBEP,'hcalTopologyConstants')
    if hasattr(process,'HcalTopologyIdealEP'):
        if hasattr(process.HcalTopologyIdealEP,'hcalTopologyConstants'):
            delattr(process.HcalTopologyIdealEP,'hcalTopologyConstants')
    if hasattr(process,'CaloTowerGeometryFromDBEP'):
        if hasattr(process.CaloTowerGeometryFromDBEP,'hcalTopologyConstants'):
            delattr(process.CaloTowerGeometryFromDBEP,'hcalTopologyConstants')
    return process

def customiseFor12346(process):
    if hasattr(process, 'hltMetCleanUsingJetID'):
       if hasattr(process.hltMetCleanUsingJetID, 'usePt'):
           delattr(process.hltMetCleanUsingJetID, 'usePt')
    return process

def customiseFor12718(process):
    for pset in process._Process__psets.values():
        if hasattr(pset,'ComponentType'):
            if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
                if not hasattr(pset,'minGoodStripCharge'):
                    pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    return process

def customiseFor12387(process):
    for producer in producers_by_type(process, 'SiStripRawToDigiModule'):
        if not hasattr(producer,'LegacyUnpacker'):
            setattr(producer,'LegacyUnpacker',cms.bool(False))
    return process

#
# CMSSW version specific customizations
def customiseHLTforCMSSW(process, menuType="GRun", fastSim=False):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_8_0":
        process = customiseFor12346(process)
        process = customiseFor11920(process)
        process = customiseFor12718(process)
        process = customiseFor12387(process)

    return process
