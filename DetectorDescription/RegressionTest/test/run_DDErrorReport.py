#! /usr/bin/env python3

import FWCore.ParameterSet.Config as cms
import sys
import os

# get the list of XML files from the cfi file
process = cms.Process("TEST")
cfiFile = 'Geometry/CMSCommonData/cmsIdealGeometryXML_cfi'
if len(sys.argv) > 1:
    cfiFile = sys.argv[1]
process.load(cfiFile)
xmlFiles = process.es_sources['XMLIdealGeometryESSource'].geomXMLFiles.value()

# create an XML configuration file that contains the same list of XML files as the python cfi file
configXMLFile = open('dddreportconfig.xml', 'w')
configXMLFile.write('<?xml version="1.0"?>\n')
configXMLFile.write('<Configuration xmlns="." xmlns:xsi="." xsi:schemaLocation= "." name="CMSConfiguration" version="0">\n')
configXMLFile.write('  <Include>\n')
for name in xmlFiles:
    configXMLFile.write("    <File name=\"" + name + "\" url=\".\"/>\n")
configXMLFile.write('  </Include>\n')
configXMLFile.write('<Root fileName="cms.xml" logicalPartName="OCMS"/>\n')
configXMLFile.write('</Configuration>\n')
configXMLFile.close()

# DDErrorReport is in $PATH
command = "DDErrorReport dddreportconfig.xml -p"
status = os.system( command )
if (status != 0):
    sys.exit(1)
