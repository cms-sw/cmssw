#! /usr/bin/env python

from __future__ import print_function
from FWCore.ParameterSet.pfnInPath import pfnInPath
import FWCore.ParameterSet.Config as cms
import sys
import os
import re

if os.getenv('LOCAL_TOP_DIR') == None:
    print("The environment variable LOCAL_TOP_DIR must be set to run this script")
    print("Usually setting it equal to the value of CMSSW_BASE will do what you want")
    print("In the context of a unit test this variable is always set automatically")
    sys.exit(1)

# get the list of XML files from the cfi file
process = cms.Process("TEST")
cfiFile = 'Geometry/CMSCommonData/cmsIdealGeometryXML_cfi'
if len(sys.argv) > 1:
    cfiFile = sys.argv[1]
process.load(cfiFile)
xmlFiles = process.es_sources['XMLIdealGeometryESSource'].geomXMLFiles.value()

def callDOMCount(schemaPath, xmlPath):
    xmlFilename = os.path.basename(xmlPath)
    xmlFile = open(xmlPath, 'r')
    tmpXMLFile = open(xmlFilename, 'w')
    # Inside each XML file, there is a path to the schema file.
    # We modify this path in a copy of the XML file for two reasons.
    # The XML file might be in a package checked out in a working release
    # area and the schema file might not be checked out or vice versa.
    # This allows DOMCount to run in spite of that. The second reason
    # is that the relative path is erroneous in many of the XML files
    # and has to be fixed.
    for line in xmlFile.readlines():
        line = line.replace("../../../../../DetectorDescription/Schema/DDLSchema.xsd",schemaPath)
        line = line.replace("../../../../DetectorDescription/Schema/DDLSchema.xsd",schemaPath)
        line = line.replace("../../../DetectorDescription/Schema/DDLSchema.xsd",schemaPath)
        line = line.replace("../../DetectorDescription/Schema/DDLSchema.xsd",schemaPath)
        line = line.replace("../DetectorDescription/Schema/DDLSchema.xsd",schemaPath)
        tmpXMLFile.write(line)
    tmpXMLFile.close()
    xmlFile.close()

    # Run DOMCount
    command = 'DOMCount -v=always -n -s -f %s' % (xmlFilename)
    os.system ( command )

    # Cleanup
    os.system ("rm %s" % (xmlFilename))

# Find the schema file
schema = pfnInPath("DetectorDescription/Schema/DDLSchema.xsd").replace('file:','')
print("schema file is:")
print(schema)
sys.stdout.flush()

# Loop over the XML files listed in the cfi file and find them
# NOTE: Now that the files are in an external package, they will
# not be in a 'LOCAL_TOP_DIR'. Checking them for each IB may not
# be needed.
#
## for name in xmlFiles:
##     fullpath = '%s/src/%s' % (os.environ['LOCAL_TOP_DIR'], name)
##     if os.path.isfile(fullpath):
##         callDOMCount(schema, fullpath)
##     else:
##         # It is an error if the file is not there but the package is
##         packageDirectory =  os.environ['LOCAL_TOP_DIR'] + '/src/' + re.split('/', name)[0] + '/' + re.split('/', name)[1]
##         if os.path.isdir(packageDirectory):
##             print 'Error, xml file not found:'
##             print fullpath
##             print 'Package is there but the xml file is not'
##             sys.stdout.flush()
##             continue

##         # if there is a base release then try to find the file there
##         fullpath = '%s/src/%s' % (os.getenv('CMSSW_RELEASE_BASE'), name)
##         if os.path.isfile(fullpath):
##              callDOMCount(schema, fullpath)
##         else:
##             print 'Error, xml file not found'
##             print name
##             sys.stdout.flush()
