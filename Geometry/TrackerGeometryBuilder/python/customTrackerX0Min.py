import FWCore.ParameterSet.Config as cms
def customise(process):
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_ExtendedX0Min_38YV0"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_GEOMETRY")
                 )
        )
    return (process)
