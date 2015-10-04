import FWCore.ParameterSet.Config as cms

## change the current default GEM geometry

## GE1/1 in 2019/2023 scenario
def custom_GE11_6partitions_v1(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v2/gem11.xml')
    return process
    
def custom_GE11_8partitions_v1(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v5/gem11.xml')
    return process

def custom_GE11_9and10partitions_v1(process):
    ## This is the default version
    return process

def custom_GE11_10partitions_v1(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v3/gem11.xml')
    return process

def custom_GE11_8and8partitions_v1(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v6/gem11.xml')
    return process

def custom_GE11_8and8partitions_v2(process):
    mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v4/gem11.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v7/gem11.xml')
    return process

### GE2/2 in 2023 scenario
def custom_GE21_v7(process) :
    geomReplace( process, 'gem11.xml','Geometry/MuonCommonData/data/v7/gem11.xml')
    geomReplace( process, 'gem21.xml','Geometry/MuonCommonData/data/v7/gem21.xml')
    geomReplace( process, 'GEMSpecs.xml','Geometry/GEMGeometryBuilder/data/v7/GEMSpecs.xml')
    return process

def custom_GE21_v7_10deg(process) :
    geomReplace( process, 'gem11.xml','Geometry/MuonCommonData/data/v7/gem11.xml')
    geomReplace( process, 'gem21.xml','Geometry/MuonCommonData/data/v7_10deg/gem21.xml')
    geomReplace( process, 'GEMSpecs.xml','Geometry/GEMGeometryBuilder/data/v7_10deg/GEMSpecs.xml')
    return process

def geomReplace(process, key, targetXML) :
    mynum=-1
    originalXML=''
    for i, xml in enumerate( process.XMLIdealGeometryESSource.geomXMLFiles) :
        if ( xml.find(key) != -1 ) :
            mynum, originalXML = i, xml
            break  ## For now, to change multiple keys is not supported.
    if ( mynum != -1 and originalXML != targetXML ) :
        print "Changing Geometry from %s to %s"%(originalXML, targetXML)
        process.XMLIdealGeometryESSource.geomXMLFiles.remove(originalXML)
        process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,targetXML)
    if ( mynum == -1) :
        print "Alert! key is not found on XMLIdealGeometryESSource"
    return process

