# this file exists to enforce dependencies for the generate2023Geometry unit test
import importlib

from Configuration.StandardSequences.GeometryConf import GeometryConf

from dict2023Geometry import *
xmls = []
for detectorVersion in detectorVersionDict.values():
    xmls.append(importlib.import_module("Geometry.CMSCommonData.cmsExtendedGeometry2023"+detectorVersion+"XML_cfi"))
