from __future__ import print_function
import FWCore.ParameterSet.Config as cms

## change the current default GEM geometry
def geomReplace(process, key, targetXML) :
    mynum=-1
    originalXML=''
    for i, xml in enumerate( process.XMLIdealGeometryESSource.geomXMLFiles) :
        if ( xml.find(key) != -1 ) :
            mynum, originalXML = i, xml
            break  ## For now, to change multiple keys is not supported.
    if ( mynum != -1 and originalXML != targetXML ) :
        print("Changing Geometry from %s to %s"%(originalXML, targetXML))
        process.XMLIdealGeometryESSource.geomXMLFiles.remove(originalXML)
        process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,targetXML)
    if ( mynum == -1) :
        print("Alert! key is not found on XMLIdealGeometryESSource")
    return process

