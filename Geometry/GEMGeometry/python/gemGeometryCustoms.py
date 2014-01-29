import FWCore.ParameterSet.Config as cms

## change the current default GEM geometry

## GE1/1 in 2019/2023 scenario
def custom_GE11_6partitions_v1(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v5/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v5/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v2/gem11.xml')
    return process
    
def custom_GE11_8partitions_v1(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v5/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v5/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v4/gem11.xml')
    return process

def custom_GE11_10partitions_v1(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v5/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v5/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v3/gem11.xml')
    return process
