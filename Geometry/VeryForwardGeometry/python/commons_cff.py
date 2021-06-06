import FWCore.ParameterSet.Config as cms
from importlib import import_module
from copy import copy

def cloneGeometry(mod_path):
    # start by importing the actual module to be cloned
    _geom = import_module(mod_path)
    # clone all geometry DDL files
    totemGeomXMLFiles = copy(_geom.totemGeomXMLFiles)
    ctppsDiamondGeomXMLFiles = copy(_geom.ctppsDiamondGeomXMLFiles)
    ctppsUFSDGeomXMLFiles = copy(_geom.ctppsUFSDGeomXMLFiles)
    ctppsPixelGeomXMLFiles = copy(_geom.ctppsPixelGeomXMLFiles)
    # clone the ESSource and ESModule to be returned
    XMLIdealGeometryESSource_CTPPS = _geom.XMLIdealGeometryESSource_CTPPS.clone()
    ctppsGeometryESModule = _geom.ctppsGeometryESModule.clone()

    return (XMLIdealGeometryESSource_CTPPS, ctppsGeometryESModule)
