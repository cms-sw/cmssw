import FWCore.ParameterSet.Config as cms
def customise(process):
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("PCastorRcd"),
                 tag = cms.string("CASTORRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PZdcRcd"),
                 tag = cms.string("ZDCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PCaloTowerRcd"),
                 tag = cms.string("CTRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalEndcapRcd"),
                 tag = cms.string("EERECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("CSCRecoDigiParametersRcd"),
                 tag = cms.string("CSCRECODIGI_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("CSCRecoGeometryRcd"),
                 tag = cms.string("CSCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalBarrelRcd"),
                 tag = cms.string("EBRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_IdealGFlash_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("IdealGFlash")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_Ideal_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("Ideal")
                 ),
        cms.PSet(record = cms.string("RPCRecoGeometryRcd"),
                 tag = cms.string("RPCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("DTRecoGeometryRcd"),
                 tag = cms.string("DTRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalPreshowerRcd"),
                 tag = cms.string("EPRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_ExtendedGFlash_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("ExtendedGFlash")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_Extended_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("Extended")
                 ),
        cms.PSet(record = cms.string("IdealGeometryRecord"),
                 tag = cms.string("TKRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PHcalRcd"),
                 tag = cms.string("HCALRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 )
        )
    return(process)

