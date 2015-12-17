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
    if hasattr(process, 'HLTPSetMuonCkfTrajectoryFilter'):
        process.HLTPSetMuonCkfTrajectoryFilter.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTIter0PSetTrajectoryFilterIT'):
        process.HLTIter0PSetTrajectoryFilterIT.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTIter1PSetTrajectoryFilterIT'):
        process.HLTIter1PSetTrajectoryFilterIT.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTIter2PSetTrajectoryFilterIT'):
        process.HLTIter2PSetTrajectoryFilterIT.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTPSetTrajectoryFilterForElectrons'):
        process.HLTPSetTrajectoryFilterForElectrons.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'process.HLTIter2HighPtTkMuPSetTrajectoryFilterIT'):
        process.HLTIter2HighPtTkMuPSetTrajectoryFilterIT.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTIter2HighPtTkMuPSetTrajectoryFilterIT'):
        process.HLTIter2HighPtTkMuPSetTrajectoryFilterIT.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTPSetMuTrackJpsiTrajectoryFilter'):
        process.HLTPSetMuTrackJpsiTrajectoryFilter.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTIter4PSetTrajectoryFilterIT'):
        process.HLTIter4PSetTrajectoryFilterIT.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    if hasattr(process, 'HLTIter3PSetTrajectoryFilterIT'):
        process.HLTIter3PSetTrajectoryFilterIT.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
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

    return process
