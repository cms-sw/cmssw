#! /usr/bin/env python

import os, sys, optparse, math, copy
from Alignment.MuonAlignment.geometryXMLparser import MuonGeometry, dtorder, cscorder

copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""
commandline = " ".join(copyargs)
print commandline

prog = sys.argv[0]

usage = """%(prog)s INPUT_XML_0  INPUT_XML_N  INPUT_REPORT_N  OUTPUT_XML  [--nsigma NSIGMA] [--dt] [--csc]

Required arguments:

INPUT_XML_0      is xml file name with chamber positions from the initial (iteration 0) geometry
INPUT_XML_N      is xml file name with chamber positions from the last iteration's geometry
INPUT_REPORT_N   is a _report.py file from the last iteration
OUTPUT_XML       is the resulting .xml file

Options:

--nsigma NSIGMA  optional minimum position displacement (measured in nsigma of deltax) in order to move a chamber
                 default NSIGMA=3
--dt             if present, DT chambers will be included
--csc            if present, CSC chambers will be included
                 NOTE: By default, having no --dt or --csc specified is equivalent to "--dt --csc"
""" % vars()

if len(sys.argv) < 5:
  print "Too few arguments.\n\n"+usage
  sys.exit()

parser=optparse.OptionParser(usage)

parser.add_option("-s","--nsigma",
  help="optional minimum nsigma(deltax) position displacement in order to move a chamber; default NSIGMA=3",
  type="int",
  default=3,
  dest="nsigma")

parser.add_option("--dt",
  help="If it is present, but not --csc, DT chambers only will be considered",
  action="store_true",
  default=False,
  dest="dt")

parser.add_option("--csc",
  help="If this is present, but not --dt, CSC chambers only will be considered",
  action="store_true",
  default=False,
  dest="csc")

options, args = parser.parse_args(sys.argv[5:])

theInXML0 = sys.argv[1]
theInXMLN = sys.argv[2]
theInREPN = sys.argv[3]
theOutXML = sys.argv[4]

theNSigma = options.nsigma

DO_DT  = False
DO_CSC = False
if options.dt or not ( options.dt or options.csc):
  DO_DT = True
if options.csc or not ( options.dt or options.csc):
  DO_CSC = True


if not os.access(theInXML0,os.F_OK): print theInXML0,"not found!\n"
if not os.access(theInXMLN,os.F_OK): print theInXMLN,"not found!\n"
if not os.access(theInREPN,os.F_OK): print theInREPN,"not found!\n"
if not (os.access(theInXML0,os.F_OK) and os.access(theInXMLN,os.F_OK) and os.access(theInREPN,os.F_OK)):
  print usage
  sys.exit()

geom0 = MuonGeometry(file(theInXML0))
geomN = MuonGeometry(file(theInXMLN))
execfile(theInREPN)

def loopover(muSystem):
  if muSystem == "DT":
    keys = geom0.dt.keys()
    keys.sort(dtorder)
  elif muSystem == "CSC":
    keys = geom0.csc.keys()
    keys.sort(cscorder)
  else: raise Exception
  
  nkeys, nkeysr, nmoved, nnotmoved = 0,0,0,0
  
  for key in keys:
    nkeys+=1
    if muSystem == "DT":
      g1 = geom0.dt[key]
      g2 = geomN.dt[key]
    else:
      g1 = geom0.csc[key]
      g2 = geomN.csc[key]
    
    if g1.relativeto != g2.relativeto:
      print "%s %s relativeto=\"%s\" versus relativeto=\"%s\"" % (muSystem, str(key), g1.relativeto, g2.relativeto)

    found = False
    #r = reports[0]
    for r in reports:
      if r.postal_address[1:] == key:
        if r.status != "PASS": break
        found = True
        rep = r
        break
    if not found:
      print muSystem, str(key), "not found. continue..."
      continue
    nkeysr+=1
    
    # check that movement is in reasonable range of 10 cm:
    if abs(g1.x - g2.x) >= 10. :
      print "Warning!!!", muSystem, str(key), "moved too much:", g1.x - g2.x, "skipping..."
      continue
    
    #print "%s %s position difference: %g - %g = %g --> %g, nsigma: %g)" % (
    #        muSystem, str(key), g1.x, g2.x, g1.x - g2.x, abs(g1.x - g2.x)/rep.deltax.error, theNSigma * rep.deltax.error)
    if abs(g1.x - g2.x) >= theNSigma * rep.deltax.error :
      print muSystem, str(key), "moved!"
      nmoved+=1
      #move this chamber:
      if muSystem == "DT":
        geom0.dt[key] = copy.copy(geomN.dt[key])
      else:
        geom0.csc[key] = copy.copy(geomN.csc[key])
    else:
      print muSystem, str(key), "not moved"
      nnotmoved+=1
      
  print muSystem,": #chambers in   geometry =", nkeys, "   PASSed report status =", nkeysr, "   moved =", nmoved, "   not moved =", nnotmoved, "   with N sigma ", theNSigma
  
  
if DO_DT: loopover("DT")
if DO_CSC: loopover("CSC")

geom0.xml(file(theOutXML, "w"))
