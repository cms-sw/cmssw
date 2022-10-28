import FWCore.ParameterSet.Config as cms
from importlib import import_module
from copy import copy

def cloneGeometry(mod_path):
    # start by importing the actual module to be cloned
    _geom = import_module(mod_path)
    # clone all geometry DDL files
    if hasattr(_geom, 'totemGeomXMLFiles'):
        totemGeomXMLFiles = copy(_geom.totemGeomXMLFiles)
    if hasattr(_geom, 'ctppsDiamondGeomXMLFiles'):
        ctppsDiamondGeomXMLFiles = copy(_geom.ctppsDiamondGeomXMLFiles)
    if hasattr(_geom, 'ctppsUFSDGeomXMLFiles'):
        ctppsUFSDGeomXMLFiles = copy(_geom.ctppsUFSDGeomXMLFiles)
    if hasattr(_geom, 'ctppsPixelGeomXMLFiles'):
        ctppsPixelGeomXMLFiles = copy(_geom.ctppsPixelGeomXMLFiles)
    # clone the ESSource and ESModule to be returned
    XMLIdealGeometryESSource_CTPPS = _geom.XMLIdealGeometryESSource_CTPPS.clone()
    ctppsGeometryESModule = _geom.ctppsGeometryESModule.clone()

    return (XMLIdealGeometryESSource_CTPPS, ctppsGeometryESModule)
