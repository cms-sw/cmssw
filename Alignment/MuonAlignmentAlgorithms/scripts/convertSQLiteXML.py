#! /usr/bin/env python

import os, sys, optparse, math

prog = sys.argv[0]

usage = """%(prog)s INPUT_FILE OUTPUT_FILE [--noChambers] [--noLayers] [--ringsOnly] [--relativeTo ideal|none]

performs either sqlite-->xml or xml-->sqlite conversion following the documentation at 
https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonGeometryConversion

Arguments:

INPUT_FILE      is either .db SQLite or .xml file that should be converted
OUTPUT_FILE     is either .xml or .db output file, the result of conversion

Options for sqlite-->xml conversion:

--noChambers    if present, no chambers info would be written into xml
--noLayers      if present, no layers (and no DT superlayers) info would be written into xml
--relativeTo X  by default, xml conversion is done relative to ideal DDD description,
                if "none" is specified, absolute positions would be written into xml
--ringsOnly     special flag for xml dumping of CSC ring structure only, it automatically
                turns off all DTs and also CSC's chambers and layers on output and coordinates
                are relative "to none"
""" % vars()

if len(sys.argv) < 3:
  print "Too few arguments!\n\n"+usage
  sys.exit()

parser=optparse.OptionParser(usage)

parser.add_option("--noChambers",
  help="if present, no chambers info would be written into xml",
  action="store_true",
  default=False,
  dest="noChambers")

parser.add_option("--noLayers",
  help="if present, no layers (and no DT superlayers) info would be written into xml",
  action="store_true",
  default=False,
  dest="noLayers")

parser.add_option("-l", "--relativeTo",
  help="by default, xml conversion is done relative to ideal DDD description, if \"none\" is specified, absolute positions would be written into xml",
  type="string",
  default='ideal',
  dest="relativeTo")

parser.add_option("--ringsOnly",
  help="special flag for xml dumping of CSC ring structure only, it automatically turns off all DTs and also CSC's chambers and layers",
  action="store_true",
  default=False,
  dest="ringsOnly")

options, args = parser.parse_args(sys.argv[3:])

supRings="True"
if options.ringsOnly: supRings="False"
supChambers="False"
if options.noChambers or options.ringsOnly: supChambers="True"
supLayers="False"
if options.noLayers or options.ringsOnly: supLayers="True"

relativeTo=options.relativeTo
if options.ringsOnly: relativeTo="none"


theInputFile = sys.argv[1]
theOutputFile = sys.argv[2]

ok = False

if theInputFile[-4:]==".xml" and theOutputFile[-3:]==".db":
  ok = True
  file("tmp_converter_cfg.py","w").write("""# xml2sqlite conversion
from Alignment.MuonAlignment.convertXMLtoSQLite_cfg import *
process.MuonGeometryDBConverter.fileName = "%(theInputFile)s"
process.PoolDBOutputService.connect = "sqlite_file:%(theOutputFile)s"

""" % vars())

if theInputFile[-3:]==".db" and theOutputFile[-4:]==".xml":
  ok = True
  file("tmp_converter_cfg.py","w").write("""# sqlite2xml conversion
from Alignment.MuonAlignment.convertSQLitetoXML_cfg import *

process.PoolDBESSource.connect = "sqlite_file:%(theInputFile)s"
process.MuonGeometryDBConverter.outputXML.fileName = "%(theOutputFile)s"

process.MuonGeometryDBConverter.outputXML.relativeto = "%(relativeTo)s"

process.MuonGeometryDBConverter.outputXML.suppressDTBarrel = True
process.MuonGeometryDBConverter.outputXML.suppressDTWheels = True
process.MuonGeometryDBConverter.outputXML.suppressDTStations = True
process.MuonGeometryDBConverter.outputXML.suppressDTChambers = %(supChambers)s
process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = %(supLayers)s
process.MuonGeometryDBConverter.outputXML.suppressDTLayers = %(supLayers)s

process.MuonGeometryDBConverter.outputXML.suppressCSCEndcaps = True
process.MuonGeometryDBConverter.outputXML.suppressCSCStations = True
process.MuonGeometryDBConverter.outputXML.suppressCSCRings = %(supRings)s
process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = %(supChambers)s
process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = %(supLayers)s

""" % vars())

if not ok:
  print usage
  sys.exit()

exit_code = os.system("cmsRun tmp_converter_cfg.py")

if exit_code>0:
  print "problem: cmsRun exited with code:", exit_code
else: 
  os.system("rm tmp_converter_cfg.py")
