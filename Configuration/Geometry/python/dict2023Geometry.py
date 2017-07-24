maxsections = 5

commonDict = {
    "abbrev" : "O",
    "name" : "common",
    "default" : 2,
    "O2" : {
        1 : [
            'Geometry/CMSCommonData/data/materials.xml',
            'Geometry/CMSCommonData/data/rotations.xml',
            'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
            'Geometry/CMSCommonData/data/cms/2023/v1/cms.xml',
            'Geometry/CMSCommonData/data/eta3/etaMax.xml',        
            'Geometry/CMSCommonData/data/cmsMother.xml',
            'Geometry/CMSCommonData/data/cmsTracker.xml',
            'Geometry/CMSCommonData/data/caloBase/2023/v1/caloBase.xml',
            'Geometry/CMSCommonData/data/cmsCalo.xml',
            'Geometry/CMSCommonData/data/muonBase/2023/v2/muonBase.xml',
            'Geometry/CMSCommonData/data/cmsMuon.xml',
            'Geometry/CMSCommonData/data/mgnt.xml',
            'Geometry/CMSCommonData/data/beampipe/2023/v1/beampipe.xml',
            'Geometry/CMSCommonData/data/cmsBeam/2023/v1/cmsBeam.xml',
            'Geometry/CMSCommonData/data/muonMB.xml',
            'Geometry/CMSCommonData/data/muonMagnet.xml',
            'Geometry/CMSCommonData/data/cavern/2017/v2/cavern.xml',
            'Geometry/CMSCommonData/data/cavernData/2017/v1/cavernData.xml',
            'Geometry/CMSCommonData/data/cavernFloor/2017/v1/cavernFloor.xml',
        ],
        5 : [
            'Geometry/CMSCommonData/data/FieldParameters.xml',
        ],
        "era" : "run2_common, phase2_common",
    }    
}

trackerDict = {
    "abbrev" : "T",
    "name" : "tracker",
    "default" : 5,
    "T4" : {
        1 : [
	    'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml',
            'Geometry/TrackerCommonData/data/pixfwdCommon.xml',
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwd.xml', 
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixbar.xml', 
            'Geometry/TrackerCommonData/data/trackermaterial.xml',
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/tracker.xml',
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixel.xml',
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/trackerbar.xml',
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/trackerfwd.xml',
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/trackerStructureTopology.xml',
            'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixelStructureTopology.xml',
            'Geometry/TrackerSimData/data/PhaseII/FlatTracker/trackersens.xml',
            'Geometry/TrackerSimData/data/PhaseII/FlatTracker/pixelsens.xml',
            'Geometry/TrackerRecoData/data/PhaseII/FlatTracker/trackerRecoMaterial.xml',
            'Geometry/TrackerSimData/data/PhaseII/FlatTracker/trackerProdCuts.xml',
            'Geometry/TrackerSimData/data/PhaseII/FlatTracker/pixelProdCuts.xml',
            'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',

        ],
        "sim" : [
            'from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *',
            'from SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkFlat_cff import *',
        ],
        "reco" : [
            'from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *',
            'from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *',
            'from Geometry.TrackerGeometryBuilder.trackerParameters_cfi import *',
            'from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *',
            'from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *',
            'trackerGeometry.applyAlignment = cms.bool(False)',
        ],
        "era" : "phase2_tracker, trackingPhase2PU140",
    },
    "T5" : {
        1 : [
            'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml',
            'Geometry/TrackerCommonData/data/pixfwdCommon.xml',
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixfwd.xml', 
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixbar.xml', 
            'Geometry/TrackerCommonData/data/trackermaterial.xml',
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/tracker.xml',
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixel.xml',
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/trackerbar.xml',
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/trackerfwd.xml',
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/trackerStructureTopology.xml',
            'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixelStructureTopology.xml',
            'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/trackersens.xml',
            'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/pixelsens.xml',
            'Geometry/TrackerRecoData/data/PhaseII/TiltedTracker4025/trackerRecoMaterial.xml',
            'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/trackerProdCuts.xml',
            'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/pixelProdCuts.xml',
            'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',
        ],
        "sim" : [
            'from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *',
            'from SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkTilted4025_cff import *',
        ],
        "reco" : [
            'from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *',
            'from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *',
            'from Geometry.TrackerGeometryBuilder.trackerParameters_cfi import *',
            'from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *',
            'from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *',
            'trackerGeometry.applyAlignment = cms.bool(False)',
        ],
        "era" : "phase2_tracker, trackingPhase2PU140",
    }
       
}

caloDict = {
    "abbrev" : "C",
    "name" : "calo",
    "default" : 3,
    "C3" : {
        1 : [
            'Geometry/EcalCommonData/data/ectkcable.xml',
            'Geometry/EcalCommonData/data/PhaseII/eregalgo.xml',
            'Geometry/EcalCommonData/data/ebalgo.xml',
            'Geometry/EcalCommonData/data/ebcon.xml',
            'Geometry/EcalCommonData/data/ebrot.xml',
            'Geometry/EcalCommonData/data/eecon.xml',
            'Geometry/EcalCommonData/data/PhaseII/escon.xml',
            'Geometry/EcalCommonData/data/PhaseII/esalgo.xml',
            'Geometry/HcalCommonData/data/hcalrotations.xml',
            'Geometry/HcalCommonData/data/hcal/HGCal/hcalalgo.xml',
            'Geometry/HcalCommonData/data/hcalbarrelalgo.xml',
            'Geometry/HcalCommonData/data/hcalendcap/SSAbsorber/hcalendcapalgo.xml',
            'Geometry/HcalCommonData/data/hcalouteralgo.xml',
            'Geometry/HcalCommonData/data/hcalforwardalgo.xml',
            'Geometry/HcalCommonData/data/hcalSimNumbering/2023/hcalSimNumbering.xml',
            'Geometry/HcalCommonData/data/hcalRecNumbering/2023/hcalRecNumbering.xml',
            'Geometry/HcalCommonData/data/average/hcalforwardmaterial.xml',
            'Geometry/HGCalCommonData/data/hgcal/v8/hgcal.xml',
            'Geometry/HGCalCommonData/data/hgcalEE/v8/hgcalEE.xml',
            'Geometry/HGCalCommonData/data/hgcalHEsil/v8/hgcalHEsil.xml',
            'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
            'Geometry/HGCalCommonData/data/hgcalCons/v8/hgcalCons.xml',
        ],
        3 : [
            'Geometry/EcalSimData/data/PhaseII/ecalsens.xml',
            'Geometry/HcalCommonData/data/hcalsens/HGCal/hcalsenspmf.xml',
            'Geometry/HcalSimData/data/hf.xml',
            'Geometry/HcalSimData/data/hfpmt.xml',
            'Geometry/HcalSimData/data/hffibrebundle.xml',
            'Geometry/HcalSimData/data/CaloUtil.xml',
            'Geometry/HGCalSimData/data/hgcsensv8.xml',
            'Geometry/HGCalSimData/data/hgccons.xml',
            'Geometry/HGCalSimData/data/hgcProdCuts.xml',
        ],
        4 : [
            'Geometry/HcalSimData/data/HcalProdCuts.xml',
            'Geometry/EcalSimData/data/EcalProdCuts.xml',
        ],
        "sim" : [
            'from Geometry.HcalCommonData.hcalParameters_cfi      import *',
            'from Geometry.HcalCommonData.hcalDDDSimConstants_cfi import *',
            'from Geometry.HGCalCommonData.hgcalV6ParametersInitialization_cfi import *',
            'from Geometry.HGCalCommonData.hgcalV6NumberingInitialization_cfi import *'
        ],
        "reco" : [
            'from Geometry.CaloEventSetup.HGCalV6Topology_cfi import *',
            'from Geometry.HGCalGeometry.HGCalV6GeometryESProducer_cfi import *',
            'from Geometry.CaloEventSetup.CaloTopology_cfi import *',
            'from Geometry.CaloEventSetup.CaloGeometryBuilder_cfi import *',
            'CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",',
            '    SelectedCalos = cms.vstring("HCAL"                   ,',
            '                                "ZDC"                    ,',
            '                                "EcalBarrel"             ,',
            '                                "TOWER"                  ,',
            '                                "HGCalEESensitive"       ,',
            '                                "HGCalHESiliconSensitive" ',
            '    )',
            ')',
            'from Geometry.EcalAlgo.EcalBarrelGeometry_cfi import *',
            'from Geometry.HcalEventSetup.HcalGeometry_cfi import *',
            'from Geometry.HcalEventSetup.CaloTowerGeometry_cfi import *',
            'from Geometry.HcalEventSetup.CaloTowerTopology_cfi import *',
            'from Geometry.HcalCommonData.hcalDDDRecConstants_cfi import *',
            'from Geometry.HcalEventSetup.hcalTopologyIdeal_cfi import *',
            'from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *',
            'from Geometry.EcalMapping.EcalMapping_cfi import *',
            'from Geometry.EcalMapping.EcalMappingRecord_cfi import *',
        ],
        "era" : "run2_HE_2017, run2_HF_2017, run2_HCAL_2017, run3_HB, phase2_hcal, phase2_hgcal, hcalHardcodeConditions, hcalSkipPacker",
    }
}

muonDict = {
    "abbrev" : "M",
    "name" : "muon",
    "default" : 2,
    "M2" : {
        1 : [
            'Geometry/MuonCommonData/data/mbCommon/2017/v2/mbCommon.xml',
            'Geometry/MuonCommonData/data/mb1/2015/v1/mb1.xml',
            'Geometry/MuonCommonData/data/mb2/2015/v1/mb2.xml',
            'Geometry/MuonCommonData/data/mb3/2015/v1/mb3.xml',
            'Geometry/MuonCommonData/data/mb4/2015/v1/mb4.xml',
            'Geometry/MuonCommonData/data/design/muonYoke.xml',
            'Geometry/MuonCommonData/data/mf/2023/v2/mf.xml',
            'Geometry/MuonCommonData/data/rpcf/2023/v1/rpcf.xml',
            'Geometry/MuonCommonData/data/gemf/TDR_BaseLine/gemf.xml',
            'Geometry/MuonCommonData/data/gem11/TDR_BaseLine/gem11.xml',
            'Geometry/MuonCommonData/data/gem21/TDR_Dev/gem21.xml',
            'Geometry/MuonCommonData/data/csc/2015/v1/csc.xml',
            'Geometry/MuonCommonData/data/mfshield/2023/v1/mfshield.xml',
            'Geometry/MuonCommonData/data/me0/TDR_Dev/me0.xml',
        ],
        2 : [
            'Geometry/MuonCommonData/data/muonNumbering/TDR_DeV/muonNumbering.xml',
        ],
        3 : [
            'Geometry/MuonSimData/data/PhaseII/ME0EtaPart/muonSens.xml',
            'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml',
            'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml',
            'Geometry/CSCGeometryBuilder/data/cscSpecs.xml',
            'Geometry/RPCGeometryBuilder/data/PhaseII/RPCSpecs.xml',
            'Geometry/GEMGeometryBuilder/data/v7/GEMSpecsFilter.xml',
            'Geometry/GEMGeometryBuilder/data/v7/GEMSpecs.xml',
        ],
        4 : [
            'Geometry/MuonSimData/data/PhaseII/muonProdCuts.xml',
        ],
        "reco" : [
            'from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *',
            'from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *',
            'from Geometry.GEMGeometryBuilder.gemGeometry_cfi import *',
            'from Geometry.GEMGeometryBuilder.me0Geometry_cfi import *',
            'from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *',
            'from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *',
        ],
        "era" : "phase2_muon, run3_GEM",
    }

}

forwardDict = {
    "abbrev" : "F",
    "name" : "forward",
    "default" : 2,
    "F2" : {
        1 : [
            'Geometry/ForwardCommonData/data/forwardshield/2017/v1/forwardshield.xml',
            'Geometry/ForwardCommonData/data/brmrotations.xml',
            'Geometry/ForwardCommonData/data/PostLS2/brm.xml',
            'Geometry/ForwardCommonData/data/zdcmaterials.xml',
            'Geometry/ForwardCommonData/data/lumimaterials.xml',
            'Geometry/ForwardCommonData/data/zdcrotations.xml',
            'Geometry/ForwardCommonData/data/lumirotations.xml',
            'Geometry/ForwardCommonData/data/zdc.xml',
            'Geometry/ForwardCommonData/data/zdclumi.xml',
            'Geometry/ForwardCommonData/data/cmszdc.xml',
        ],
        3 : [
            'Geometry/ForwardCommonData/data/brmsens.xml',
            'Geometry/ForwardSimData/data/zdcsens.xml',
        ],
        4 : [
            'Geometry/ForwardSimData/data/zdcProdCuts.xml',
            'Geometry/ForwardSimData/data/ForwardShieldProdCuts.xml',
        ],
        "reco" :[
            'from Geometry.ForwardGeometry.ForwardGeometry_cfi import *',
        ]
    }    
}

timingDict = {
    "abbrev" : "I",
    "name" : "timing",
    "default" : 1,
    "I1" : {},
    "I2" : {
        1 : [
            'Geometry/HGCalCommonData/data/fastTimingBarrel.xml',
            'Geometry/HGCalCommonData/data/fastTimingEndcap.xml',
            'Geometry/HGCalCommonData/data/fastTimingElement.xml',
            ],
        3 : [
            'Geometry/HGCalSimData/data/fasttimesens.xml'
            ],
        4 : [
            'Geometry/HGCalSimData/data/fasttimeProdCuts.xml'
            ],
        "sim" : [
            'from Geometry.HGCalCommonData.fastTimeParametersInitialization_cfi import *',
            'from Geometry.HGCalCommonData.fastTimeNumberingInitialization_cfi import *',
        ],
        "reco" :[
            'from Geometry.CaloEventSetup.FastTimeTopology_cfi import *',
            'from Geometry.HGCalGeometry.FastTimeGeometryESProducer_cfi import *',
        ],
        "era" : "phase2_timing, phase2_timing_layer",
    }
}

allDicts = [ commonDict, trackerDict, caloDict, muonDict, forwardDict, timingDict ]

detectorVersionDict = {
    ("O2","T5","C3","M2","F2","I1") : "D17",
    ("O2","T5","C3","M2","F2","I2") : "D19",
    ("O2","T4","C3","M2","F2","I1") : "D20",
}

deprecatedDets = set([ "D1", "D2", "D3", "D5", "D6" , "D7", "D4", "D8" , "D9", "D12", "D13", "D15", "D10", "D11", "D14", "D16", "D18" ])
deprecatedSubdets = set([ "T1", "T2" ,"T3", "C1", "C2", "M1", "O1", "F1" ])
