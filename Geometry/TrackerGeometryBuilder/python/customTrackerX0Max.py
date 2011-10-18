import FWCore.ParameterSet.Config as cms
def customise(process):
    process.XMLFromDBSource.label='ExtendedX0Max'
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_311YV1_ExtendedX0Max_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("ExtendedX0Max")
                 )
        )
    return (process)
