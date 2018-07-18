#!/usr/bin/env python

import sys, optparse, re, math
from Alignment.MuonAlignment.geometryXMLparser import MuonGeometry

usage = """./%prog GeomGlobal.xml AbsConstrints.[phiy|phipos|phiz] frameName

Calculates constraints from AbsConstraints relative to GeomGlobal for
use in CSCOverlapsAlignmentAlgorithm.  Assumes that constraints (including
chamber names) are properly formatted.

GeomGlobal     CSC geometry represented as a relativeto="none" XML file
               (standard format--- must be MuonGeometryDBConverter output)
AbsConstrints  text file listing absolute phiy, phipos, or phiz constraints
               as "chambername measuredvalue uncertainty"
frameName      name of coordinate system frame (all constraints are
               relative to this coordinate system"""

parser = optparse.OptionParser(usage)
parser.add_option("--scaleErrors",
                  help="factor to scale errors: 1 is default, 10 *weakens* constraint by a factor of 10, etc.",
                  type="string",
                  default=1,
                  dest="scaleErrors")
parser.add_option("--disks",
                  help="align whole disks, rather than individual rings",
                  action="store_true",
                  dest="disks")

if len(sys.argv) < 4:
    raise SystemError("Too few arguments.\n\n"+parser.format_help())

if sys.argv[2][-5:] == ".phiy": mode = "phiy"
elif sys.argv[2][-7:] == ".phipos": mode = "phipos"
elif sys.argv[2][-5:] == ".phiz": mode = "phiz"
else: raise Exception

geometry = MuonGeometry(sys.argv[1])
constraints = file(sys.argv[2])
frameName = sys.argv[3]

options, args = parser.parse_args(sys.argv[4:])
options.scaleErrors = float(options.scaleErrors)

empty = True
byRing = {"ME+1/1": [], "ME+1/2": [], "ME+2/1": [], "ME+2/2": [], "ME+3/1": [], "ME+3/2": [], "ME+4/1": [], "ME+4/2": [], "ME-1/1": [], "ME-1/2": [], "ME-2/1": [], "ME-2/2": [], "ME-3/1": [], "ME-3/2": [], "ME-4/1": [], "ME-4/2": []}
if options.disks: byRing = {"ME+1/1": [], "YE+1": [], "YE+2": [], "ME+4/1": [], "ME+4/2": [], "ME-1/1": [], "YE-1": [], "YE-2": [], "ME-4/1": [], "ME-4/2": []}

for line in constraints.readlines():
    match = re.match(r"(ME[\+\-/0-9]+)\s+([\+\-\.eE0-9]+)\s+([\+\-\.eE0-9]+)", line)
    if match is not None:
        chamber, value, error = match.groups()
        ringName = chamber[0:6]
        value = float(value)
        error = float(error) * options.scaleErrors

        if options.disks:
            if ringName in ("ME+1/2", "ME+1/3"): ringName = "YE+1"
            elif ringName in ("ME+2/1", "ME+2/2", "ME+3/1", "ME+3/2"): ringName = "YE+2"
            elif ringName in ("ME-1/2", "ME-1/3"): ringName = "YE-1"
            elif ringName in ("ME-2/1", "ME-2/2", "ME-3/1", "ME-3/2"): ringName = "YE-2"

        if chamber[2] == "+": endcap = 1
        elif chamber[2] == "-": endcap = 2
        else: raise Exception
        station = int(chamber[3])
        ring = int(chamber[5])
        cham = int(chamber[7:9])

        if mode == "phiy":
            geom = geometry.csc[endcap, station, ring, cham].phiy
        elif mode == "phipos":
            geom = math.atan2(geometry.csc[endcap, station, ring, cham].y, geometry.csc[endcap, station, ring, cham].x)
        elif mode == "phiz":
            geom = geometry.csc[endcap, station, ring, cham].phiz

        relative = value - geom

        if mode in ("phiy", "phipos", "phiz"):
            while relative > math.pi: relative -= 2.*math.pi
            while relative <= -math.pi: relative += 2.*math.pi

        if ringName in byRing:
            byRing[ringName].append("""cms.PSet(i = cms.string("%(frameName)s"), j = cms.string("%(chamber)s"), value = cms.double(%(relative)g), error = cms.double(%(error)g))""" % vars())
            empty = False

if not empty:
    keys = sorted(byRing.keys())
    print "for fitter in process.looper.algoConfig.fitters:"
    for ringName in keys:
        if len(byRing[ringName]) > 0:
            print "    if fitter.name.value() == \"%(ringName)s\":" % vars()
            print "        fitter.alignables.append(\"%(frameName)s\")" % vars()
            for line in byRing[ringName]:
                print "        fitter.constraints.append(%(line)s)" % vars()
