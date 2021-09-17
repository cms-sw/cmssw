import FWCore.ParameterSet.Config as cms

process = cms.Process("cosmicMonitorTest")
#    include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cff" 
#    include "Geometry/CommonDetUnit/data/globalTrackingGeometry.cfi"
#    include "RecoMuon/DetLayers/data/muonDetLayerGeometry.cfi"
#    include "Configuration/GlobalRuns/data/ForceZeroTeslaField.cff"
# reconstruction sequence for Global Run
process.load("Configuration.GlobalRuns.ReconstructionGR_cff")

process.load("DQMOffline.Muon.muonCosmicMonitors_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cff")

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# Magnetic fiuld: force mag field to be 0.0 tesla
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_0T_cff")

process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")

process.load("Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/00AE5571-7C2F-DD11-88EC-000423D174FE.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/0206C353-8C2F-DD11-B204-001617C3B5E4.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/08CC3DCE-8E2F-DD11-864F-001D09F24F1F.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/0C3918D8-8E2F-DD11-ACD3-000423D992A4.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/1E41CC87-8E2F-DD11-B72A-001D09F29114.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/2076FB50-912F-DD11-B55B-001D09F292D1.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/222D6C31-7D2F-DD11-AF95-000423D9863C.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/22761350-912F-DD11-A64D-001D09F24EAC.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/24CECB60-912F-DD11-A1CB-001D09F2905B.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/2886FD98-8E2F-DD11-9BA8-000E0C3F0E47.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/301F059B-8E2F-DD11-B61C-001D09F23E53.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/30666D66-7D2F-DD11-9194-001D09F28EA3.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/38CA9370-7C2F-DD11-85A8-001617E30D06.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/38EF2045-642F-DD11-A0E2-001617E30D40.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/58EC2104-8C2F-DD11-BBA1-001617E30D12.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/5CD4EEB9-632F-DD11-AACC-000423D94700.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/6A9E2B78-7C2F-DD11-9FDC-001617C3B66C.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/820B1C66-7D2F-DD11-9CB7-001617DC1F70.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/86671C9F-8E2F-DD11-B172-000423D944F0.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/887A7B14-7D2F-DD11-9D89-0019B9F704D6.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/A2BCD751-912F-DD11-A398-001D09F2AD84.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/A87AA293-7A2F-DD11-920C-0019DB2F3F9B.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/AA918954-8C2F-DD11-AF17-000423D99658.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/B22A278C-8E2F-DD11-8F61-0019B9F72CC2.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/B6966EFF-7C2F-DD11-BFB5-0030487C608C.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/BE9CC05D-912F-DD11-9EE6-001D09F24493.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/C8125756-8C2F-DD11-B140-001D09F23A84.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/C88EE979-7D2F-DD11-A8F0-001617DBD316.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/CA7881F3-8E2F-DD11-9C13-001D09F297EF.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/CCF71B75-7E2F-DD11-807F-001D09F251D1.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/D0E86258-8C2F-DD11-A9F5-001D09F295FB.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/D28C7DB7-7B2F-DD11-A8A4-001D09F26509.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/EA22D96A-8C2F-DD11-99E6-000423D98920.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/EA8C2F00-7D2F-DD11-AAF2-001617C3B6C6.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0000/FC2CCF6F-7A2F-DD11-9853-001D09F24763.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/1C380537-932F-DD11-ADA5-001D09F242EF.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/1EC39588-A62F-DD11-A19C-001D09F251BD.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/2C4E1B6D-AD2F-DD11-9174-001D09F290BF.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/5440B530-952F-DD11-9A3B-000423D992A4.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/688B92BC-AE2F-DD11-88D9-000423D98B5C.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/A85B9849-932F-DD11-8771-000423D6C8E6.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/DEC3EA3A-932F-DD11-9FD0-001617C3B76E.root', 
        '/store/data/CRUZET1/Cosmics/RECO/CRUZET_V3_v1/0001/FE2D8873-972F-DD11-9B1D-000423D98B28.root')
)

process.prefer("GlobalTag")
process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('/tmp/reco-gr-dqm.root')
)

process.p = cms.Path(process.muonCosmicMonitors*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.GlobalTag.globaltag = 'CRUZET_V3::All'


