import FWCore.ParameterSet.Config as cms

# upgrade RecoTrackSelector to allow BTV-like cuts (PR #8679)
def customiseFor8679(process):
    if hasattr(process,'hltBSoftMuonMu5L3') :
       delattr(process.hltBSoftMuonMu5L3,'min3DHit')
       setattr(process.hltBSoftMuonMu5L3,'minLayer', cms.int32(0))
       setattr(process.hltBSoftMuonMu5L3,'min3DLayer', cms.int32(0))
       setattr(process.hltBSoftMuonMu5L3,'minPixelHit', cms.int32(0))
       setattr(process.hltBSoftMuonMu5L3,'usePV', cms.bool(False))
       setattr(process.hltBSoftMuonMu5L3,'vertexTag', cms.InputTag(''))
    return process

# Simplified TrackerTopologyEP config (PR #7966)
def customiseFor7966(process):
    if hasattr(process, 'trackerTopology'):
        params = process.trackerTopology.parameterNames_()
        for param in params:
            delattr(process.trackerTopology, param)
        setattr(process.trackerTopology, 'appendToDataLabel', cms.string(""))
    return process


# Removal of 'upgradeGeometry' from TrackerDigiGeometryESModule (PR #7794)
def customiseFor7794(process):
    if hasattr(process, 'TrackerDigiGeometryESModule'):
        if hasattr(process.TrackerDigiGeometryESModule, 'trackerGeometryConstants'):
            if hasattr(process.TrackerDigiGeometryESModule.trackerGeometryConstants, 'upgradeGeometry'):
                delattr(process.TrackerDigiGeometryESModule.trackerGeometryConstants, 'upgradeGeometry')
    return process


# CMSSW version specific customizations
def customiseHLTforCMSSW(process,menuType="GRun",fastSim=False):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_7_5":
        process = customiseFor8679(process)
        process = customiseFor7966(process)
        process = customiseFor7794(process)

    return process
