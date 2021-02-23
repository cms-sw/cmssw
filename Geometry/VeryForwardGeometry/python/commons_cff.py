import FWCore.ParameterSet.Config as cms
from importlib import import_module

def cloneGeometry(mod_path):
    _geom = import_module(mod_path)
    totemGeomXMLFiles = _geom.totemGeomXMLFiles.clone()
    ctppsDiamondGeomXMLFiles = _geom.ctppsDiamondGeomXMLFiles.clone()
    ctppsUFSDGeomXMLFiles = _geom.ctppsUFSDGeomXMLFiles.clone()
    ctppsPixelGeomXMLFiles = _geom.ctppsPixelGeomXMLFiles.clone()
    XMLIdealGeometryESSource_CTPPS = _geom.XMLIdealGeometryESSource_CTPPS.clone(
        geomXMLFiles = totemGeomXMLFiles + ctppsDiamondGeomXMLFiles + ctppsUFSDGeomXMLFiles + ctppsPixelGeomXMLFiles
    )
    ctppsGeometryESModule = _geom.ctppsGeometryESModule.clone()

    return (XMLIdealGeometryESSource_CTPPS, ctppsGeometryESModule)
