#! /usr/bin/env python

import os, sys, optparse, math

copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""
    if copyargs[i].find(" ") != -1:
        copyargs[i] = "\"%s\"" % copyargs[i]
commandline = " ".join(copyargs)

usage = """./%prog DIRNAME INITIALGEOM INPUTFILES [options]

Creates (overwrites) a directory for the jobs and creates (overwrites) submitJobs.sh with the submission sequence and dependencies.

DIRNAME        directory will be named DIRNAME
INITIALGEOM    SQLite file containing muon geometry with tag names CSCAlignmentRcd, CSCAlignmentErrorRcd
INPUTFILES     Python file defining 'fileNames', a list of input files as strings (create with findQualityFiles.py)"""

parser = optparse.OptionParser(usage)
parser.add_option("--minhits",
                  help="Minimum number of hits per chamber",
                  type="int",
                  default=6,
                  dest="minhits")
parser.add_option("--mintracks",
                  help="Minimum number of tracks per chamber",
                  type="int",
                  default=10,
                  dest="mintracks")
parser.add_option("--sequence",
                  help="Sequence of alignment procedures",
                  type="string",
                  default="phiy rphi phiz",
                  dest="sequence")
parser.add_option("--combineME11",
                  help="if invoked, combine ME1/1a and ME1/1b into rigid bodies",
                  action="store_true",
                  dest="combineME11")

if len(sys.argv) < 4:
    raise SystemError, "Too few arguments.\n\n"+parser.format_help()

DIRNAME = sys.argv[1]
INITIALGEOM = sys.argv[2]
INPUTFILES = sys.argv[3]
execfile(INPUTFILES)  # defines fileNames

options, args = parser.parse_args(sys.argv[4:])
minhits = options.minhits
mintracks = options.mintracks
sequence_conversion = {"phiy": "roty", "rphi": "phipos", "phiz": "rotz"}
sequence = map(lambda s: sequence_conversion[s], options.sequence.split(" "))
combineME11 = options.combineME11

os.system("rm -rf %s; mkdir %s" % (DIRNAME, DIRNAME))
pwd = str(os.getcwdu())
os.system("cp beamHalo_cfg.py %s/" % DIRNAME)
os.system("ln -s ../inertGlobalPositionRcd.db %s/inertGlobalPositionRcd.db" % DIRNAME)

file("%s/convertToXML_cfg.py" % DIRNAME, "w").write("""import os

fileName = os.getenv("ALIGNMENT_CONVERTXML")

from Alignment.MuonAlignment.convertSQLitetoXML_cfg import *
process.PoolDBESSource.connect = "sqlite_file:%s" % fileName
process.MuonGeometryDBConverter.outputXML.fileName = "%s.xml" % fileName[:-3]
process.MuonGeometryDBConverter.outputXML.relativeto = "ideal"
process.MuonGeometryDBConverter.outputXML.suppressDTChambers = True
process.MuonGeometryDBConverter.outputXML.suppressDTSuperLayers = True
process.MuonGeometryDBConverter.outputXML.suppressDTLayers = True
process.MuonGeometryDBConverter.outputXML.suppressCSCChambers = False
process.MuonGeometryDBConverter.outputXML.suppressCSCLayers = True

process.MuonGeometryDBConverter.getAPEs = cms.bool(True)
process.PoolDBESSource.toGet = cms.VPSet(
    cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
    cms.PSet(record = cms.string("DTAlignmentErrorRcd"), tag = cms.string("DTAlignmentErrorRcd")),
    cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
    cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd")),
      )
""")

controller = ["""#!/bin/sh
# %(commandline)s

cd %(pwd)s
eval `scramv1 run -sh`
cd %(DIRNAME)s""" % vars()]

for iteration, mode in enumerate(sequence):
    iteration += 1

    if mode == "roty": params = "000010"
    if mode == "phipos": params = "110001"
    if mode == "rotz": params = "000001"

    if iteration == 1: inputdb = "../%s" % INITIALGEOM
    else: inputdb = "%s_%02d.db" % (DIRNAME, iteration - 1)

    inputfiles = " ".join(fileNames)

    controller.append("""
export ALIGNMENT_INPUTFILES='%(inputfiles)s'
export ALIGNMENT_INPUTDB=%(inputdb)s
export ALIGNMENT_ITERATION=%(iteration)d
export ALIGNMENT_DIRNAME=%(DIRNAME)s
export ALIGNMENT_MODE=%(mode)s
export ALIGNMENT_PARAMS=%(params)s
export ALIGNMENT_MINHITS=%(minhits)d
export ALIGNMENT_MINTRACKS=%(mintracks)d
export ALIGNMENT_COMBINEME11=%(combineME11)s
cmsRun beamHalo_cfg.py | tee %(DIRNAME)s_%(iteration)02d.log
export ALIGNMENT_CONVERTXML=%(DIRNAME)s_%(iteration)02d.db
cmsRun convertToXML_cfg.py""" % vars())

controller.append("")

file("%s/align.sh" % DIRNAME, "w").write("\n".join(controller))
os.system("chmod +x %s/align.sh" % DIRNAME)

