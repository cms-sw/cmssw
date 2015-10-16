import FWCore.ParameterSet.Config as cms

# Update to replace old jet corrector mechanism
from HLTrigger.Configuration.customizeHLTforNewJetCorrectors import customizeHLTforNewJetCorrectors

# Possibility to put different ring dependent cut on ADC (PR #9232)                                                              
def customiseFor9232(process):
    if hasattr(process,'hltEcalPhiSymFilter'):
        if hasattr(process.hltEcalPhiSymFilter,'ampCut_barrel'):
            delattr(process.hltEcalPhiSymFilter,'ampCut_barrel')
        if hasattr(process.hltEcalPhiSymFilter,'ampCut_endcap'):
            delattr(process.hltEcalPhiSymFilter,'ampCut_endcap')
    return process

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


# Removal of L1 Stage 1 unpacker configuration from config (PR #10087)
def customiseFor10087(process):
    if hasattr(process, 'hltCaloStage1Digis'):
        if hasattr(process.hltCaloStage1Digis, 'FWId'):
            delattr(process.hltCaloStage1Digis, 'FWId')
        if hasattr(process.hltCaloStage1Digis, 'FedId'):
            delattr(process.hltCaloStage1Digis, 'FedId')
    return process

def customiseFor10234(process):
    if hasattr(process, 'hltCaloStage1Digis'):
        if hasattr(process.hltCaloStage1Digis, 'FWId'):
            delattr(process.hltCaloStage1Digis, 'FWId')
        if hasattr(process.hltCaloStage1Digis, 'FedId'):
            delattr(process.hltCaloStage1Digis, 'FedId')
    return process

def customiseFor10353(process):
    # Take care of geometry changes in HCAL
    if not hasattr(process,'hcalDDDSimConstants'):
        process.hcalDDDSimConstants = cms.ESProducer( 'HcalDDDSimConstantsESModule' )
    if not hasattr(process,'hcalDDDRecConstants'):
        process.hcalDDDRecConstants = cms.ESProducer( 'HcalDDDRecConstantsESModule' )
    return process

# upgrade RecoTrackSelector to allow selection on originalAlgo (PR #10418)
def customiseFor10418(process):
    if hasattr(process,'hltBSoftMuonMu5L3') :
       setattr(process.hltBSoftMuonMu5L3,'originalAlgorithm', cms.vstring())
       setattr(process.hltBSoftMuonMu5L3,'algorithmMaskContains', cms.vstring())
    return process

# migrate RPCPointProducer to a global::EDProducer (PR #10927)
def customiseFor10927(process):
    if any(module.type_() is 'RPCPointProducer' for module in process.producers.itervalues()):
        if not hasattr(process, 'CSCObjectMapESProducer'):
            process.CSCObjectMapESProducer = cms.ESProducer( 'CSCObjectMapESProducer' )
        if not hasattr(process, 'DTObjectMapESProducer'):
            process.DTObjectMapESProducer = cms.ESProducer( 'DTObjectMapESProducer' )
    return process

# change RecoTrackRefSelector to stream::EDProducer (PR #10911)
def customiseFor10911(process):
    if hasattr(process,'hltBSoftMuonMu5L3'):
        # Switch module type from EDFilter to EDProducer
        process.hltBSoftMuonMu5L3 = cms.EDProducer("RecoTrackRefSelector", **process.hltBSoftMuonMu5L3.parameters_())
    return process

# Fix MeasurementTrackerEvent configuration in several TrackingRegionProducers (PR 11183)
def customiseFor11183(process):
    def useMTEName(componentName):
        if componentName in ["CandidateSeededTrackingRegionsProducer", "TrackingRegionsFromBeamSpotAndL2Tau"]:
            return "whereToUseMeasurementTracker"
        return "howToUseMeasurementTracker"

    def replaceInPSet(pset, moduleLabel):
        for paramName in pset.parameterNames_():
            param = getattr(pset, paramName)
            if isinstance(param, cms.PSet):
                if hasattr(param, "ComponentName") and param.ComponentName.value() in ["CandidateSeededTrackingRegionsProducer", "TauRegionalPixelSeedGenerator"]:
                    useMTE = useMTEName(param.ComponentName.value())

                    if hasattr(param.RegionPSet, "measurementTrackerName"):
                        param.RegionPSet.measurementTrackerName = cms.InputTag(param.RegionPSet.measurementTrackerName.value())
                        if hasattr(param.RegionPSet, useMTE):
                            raise Exception("Assumption of CandidateSeededTrackingRegionsProducer not having '%s' parameter failed" % useMTE)
                        setattr(param.RegionPSet, useMTE, cms.string("ForSiStrips"))
                    else:
                        setattr(param.RegionPSet, useMTE, cms.string("Never"))
                else:
                    replaceInPSet(param, moduleLabel)
            elif isinstance(param, cms.VPSet):
                for element in param:
                    replaceInPSet(element, moduleLabel)

    for label, module in process.producers_().iteritems():
        replaceInPSet(module, label)

    return process

# CMSSW version specific customizations
def customiseHLTforCMSSW(process, menuType="GRun", fastSim=False):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_7_6":
        process = customiseFor10418(process)
        process = customiseFor10353(process)
        process = customiseFor10911(process)
        process = customiseFor11183(process)
    if cmsswVersion >= "CMSSW_7_5":
        process = customiseFor10927(process)
        process = customiseFor9232(process)
        process = customiseFor8679(process)
        process = customiseFor8356(process)
        process = customiseFor7966(process)
        process = customiseFor7794(process)
        process = customiseFor10087(process)
        process = customizeHLTforNewJetCorrectors(process)
    if cmsswVersion >= "CMSSW_7_4":
        process = customiseFor10234(process)

    return process
