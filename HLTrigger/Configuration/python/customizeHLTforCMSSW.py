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


# Updating the config (PR #8356)
def customiseFor8356(process):
    MTRBPSet = cms.PSet(
        Rescale_eta = cms.double( 3.0 ),
        Rescale_phi = cms.double( 3.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        UseVertex = cms.bool( False ),
        Pt_fixed = cms.bool( False ),
        Z_fixed = cms.bool( True ),
        Phi_fixed = cms.bool( False ),
        Eta_fixed = cms.bool( False ),
        Pt_min = cms.double( 1.5 ),
        Phi_min = cms.double( 0.1 ),
        Eta_min = cms.double( 0.1 ),
        DeltaZ = cms.double( 15.9 ),
        DeltaR = cms.double( 0.2 ),
        DeltaEta = cms.double( 0.2 ),
        DeltaPhi = cms.double( 0.2 ),
        maxRegions = cms.int32( 2 ),
        precise = cms.bool( True ),
        OnDemand = cms.int32( -1 ),
        MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
    )

    def producers_by_type(process, type):
    	return (module for module in process._Process__producers.values() if module._TypedParameterizable__type == type)

    for l3MPModule in producers_by_type(process, 'L3MuonProducer'):
	if hasattr(l3MPModule, 'GlbRefitterParameters'):
            l3MPModule.GlbRefitterParameters.RefitFlag = cms.bool(True)
        if hasattr(l3MPModule, 'L3TrajBuilderParameters'):
            if hasattr(l3MPModule.L3TrajBuilderParameters, 'MuonTrackingRegionBuilder'):
                l3MPModule.L3TrajBuilderParameters.MuonTrackingRegionBuilder = MTRBPSet

    listL3seedingModule = ['hltL3TrajSeedIOHit','hltL3NoFiltersNoVtxTrajSeedIOHit','hltHIL3TrajSeedIOHit']
    for l3IOTrajModule in listL3seedingModule:
	if hasattr(process, l3IOTrajModule):
	    if hasattr(getattr(process, l3IOTrajModule), 'MuonTrackingRegionBuilder'):
                setattr(getattr(process, l3IOTrajModule), 'MuonTrackingRegionBuilder', MTRBPSet)

    return process


# Simplified TrackerTopologyEP config (PR #7966)
def customiseFor7966(process):
    if hasattr(process, 'trackerTopology'):
        params = process.trackerTopology.parameterNames_()
        for param in params:
            delattr(process.trackerTopology, param)
        setattr(process.trackerTopology, 'appendToDataLabel', cms.string(""))
    if hasattr(process,'TrackerDigiGeometryESModule'):
        if hasattr(process.TrackerDigiGeometryESModule,'trackerGeometryConstants'):
            delattr(process.TrackerDigiGeometryESModule,'trackerGeometryConstants')
    return process

# Removal of 'upgradeGeometry' from TrackerDigiGeometryESModule (PR #7794)
def customiseFor7794(process):
    if hasattr(process, 'TrackerDigiGeometryESModule'):
        if hasattr(process.TrackerDigiGeometryESModule, 'trackerGeometryConstants'):
            if hasattr(process.TrackerDigiGeometryESModule.trackerGeometryConstants, 'upgradeGeometry'):
                delattr(process.TrackerDigiGeometryESModule.trackerGeometryConstants, 'upgradeGeometry')
    return process


# Removal of L1 Stage 1 unpacker configuration from config (PR #10234 aka 10163/10087)
def customiseFor10234(process):
    if hasattr(process, 'hltCaloStage1Digis'):
        if hasattr(process.hltCaloStage1Digis, 'FWId'):
            delattr(process.hltCaloStage1Digis, 'FWId')
        if hasattr(process.hltCaloStage1Digis, 'FedId'):
            delattr(process.hltCaloStage1Digis, 'FedId')
    return process

# CMSSW version specific customizations
def customiseHLTforCMSSW(process,menuType="GRun",fastSim=False):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_7_5":
        process = customiseFor8679(process)
        process = customiseFor8356(process)
        process = customiseFor7966(process)
        process = customiseFor7794(process)
    if cmsswVersion >= "CMSSW_7_4":
        process = customiseFor10234(process)
    return process
