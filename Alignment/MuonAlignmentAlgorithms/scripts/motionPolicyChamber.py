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
  default=2,
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

  movedChamberKeys = []
  
  if muSystem == "DT":
    keys = geom0.dt.keys()
    keys.sort(dtorder)
  elif muSystem == "CSC":
    keys = geom0.csc.keys()
    keys.sort(cscorder)
  else: raise Exception
  
  nkeys, nkeysr, nkeyspass, nmoved, nnotmoved = 0,0,0,0,0
  nfail_toideal, nfail_deltafinal, nfail_lowstat, nfail_nsigma = 0,0,0,0
  nok_toideal, nok_deltafinal, nok_lowstat, nok_nsigma = 0,0,0,0
  
  for key in keys:
    is_ch = True
    if muSystem == "DT":
      if len(key) != 3: is_ch = False
      ch_key = key[:3]
      g1 = geom0.dt[key]
      g2 = geomN.dt[key]
      ch_g1 = geom0.dt[ch_key]
      ch_g2 = geomN.dt[ch_key]
    else:
      if len(key) != 4: is_ch = False
      ch_key = key[:4]
      g1 = geom0.csc[key]
      g2 = geomN.csc[key]
      ch_g1 = geom0.csc[ch_key]
      ch_g2 = geomN.csc[ch_key]
    if is_ch: nkeys+=1

    chWasntMoved = True
    if ch_key in movedChamberKeys:
      chWasntMoved = False
    
    if g1.relativeto != g2.relativeto:
      print "%s %s relativeto=\"%s\" versus relativeto=\"%s\"" % (muSystem, str(key), g1.relativeto, g2.relativeto)

    found = False
    #r = reports[0]
    for r in reports:
      if r.postal_address[1:] == ch_key:
        found = True
        rep = r
        break
    if not found:
      if is_ch: print muSystem, str(key), "not found in the report. Continue..."
      continue
    if is_ch: nkeysr+=1

    if rep.status != "PASS":
      if is_ch: print muSystem, str(key), "status is not PASS: %s   Continue..." % rep.status
      continue
    #print muSystem, str(key), str(ch_key)
    if is_ch: nkeyspass+=1
    
    ############################################################
    # CHECKS
    
    ok = True
    
    if muSystem == "DT" and chWasntMoved:
      
      # check that chamber's movement respective to ideal geometry is in reasonable range of 20 mm or 20 mrad:
      if abs(ch_g2.x) > 2. or abs(ch_g2.y) > 2. or abs(ch_g2.phiy) > 0.02 or abs(ch_g2.phiz) > 0.02:
        ok = False
        if is_ch:
          nfail_toideal += 1
          print "Warning!!!", muSystem, str(key), \
            "moved too much with respect to ideal: dx=%.2f mm  dy=%.2f mm  dphiy=%.2f mrad  dphiz=%.2f mrad  skipping..." % (ch_g2.x*10, ch_g2.y*10, ch_g2.phiy*1000, ch_g2.phiz*1000)
      if is_ch and ok: nok_toideal +=1
      
      # check that movements during the final iteration were negligible
      # separately for station 4
      if key[1] != 4 :
        if abs(rep.deltax.value) > 0.03 or abs(rep.deltay.value) > 0.03 or abs(rep.deltaphiy.value) > 0.0003 or abs(rep.deltaphiz.value) > 0.0006:
          ok = False
          if is_ch:
            nfail_deltafinal += 1
            print "Warning!!!", muSystem, str(key), \
              "moved too much at final iteration: dx=%.2f mm  dy=%.2f mm  dphiy=%.2f mrad  dphiz=%.2f mrad   skipping..." % \
                (rep.deltax.value*10, rep.deltay.value*10, rep.deltaphiy.value*1000, rep.deltaphiz.value*1000)
      else:
        if abs(rep.deltax.value) > 0.03 or abs(rep.deltaphiy.value) > 0.0003 or abs(rep.deltaphiz.value) > 0.0006:
          ok = False
          if is_ch:
            nfail_deltafinal += 1
            print "Warning!!!", muSystem, str(key), \
              "moved too much at final iteration: dx=%.2f mm  dphiy=%.2f mrad  dphiz=%.2f mrad   skipping..." % \
                (rep.deltax.value*10, rep.deltaphiy.value*1000, rep.deltaphiz.value*1000)
      if is_ch and ok: nok_deltafinal +=1

      # low statictics check:
      if rep.deltax.error > 0.5:
        ok = False
        if is_ch:
          nfail_lowstat +=1
          print "Warning!!!", muSystem, str(key), "low statistics chamber with too big dx.error = %.2f mm   skipping..." % (rep.deltax.error*10.)
      if is_ch and ok: nok_lowstat +=1
    
      # N-SIGMA MOTION POLICY CHECK
      #print "%s %s position difference: %g - %g = %g --> %g, nsigma: %g)" % (
      #        muSystem, str(key), g1.x, g2.x, g1.x - g2.x, abs(g1.x - g2.x)/rep.deltax.error, theNSigma * rep.deltax.error)
      if abs(ch_g1.x - ch_g2.x) < theNSigma * math.sqrt ( rep.deltax.error*rep.deltax.error + 0.02*0.02 ) :
        ok = False
        if is_ch:
          nfail_nsigma += 1
          print muSystem, str(key), "not moved: xN-x0 = %.3f - %.3f = %.3f < %.3f mm" % \
            ( ch_g2.x*10., ch_g1.x*10., (ch_g2.x-ch_g1.x)*10., theNSigma * math.sqrt ( rep.deltax.error*rep.deltax.error + 0.02*0.02 )*10.)

      if ok: chWasntMoved = False
    
    if not ok or chWasntMoved: continue
    
    ############################################################
    # MOTION
    if is_ch:
      movedChamberKeys.append(ch_key)
      print muSystem, str(key), "moved!"
      nmoved+=1
    #move this chamber/superlayer/layer:
    if muSystem == "DT":
      geom0.dt[key] = copy.copy(geomN.dt[key])
    else:
      geom0.csc[key] = copy.copy(geomN.csc[key])
    
  nnotmoved = nkeys - nmoved
  nsig = int(theNSigma)
  print """
%(muSystem)s REPORT for  %(nsig)d sigma policy:
Cumulative counters:
  %(nkeys)d\t chambers in geometry
  %(nkeysr)d\t chambers in report
  %(nkeyspass)d\t have PASS status
  %(nok_toideal)d\t pass big shift with respect to ideal check
  %(nok_deltafinal)d\t pass big deltas during final iteration
  %(nok_lowstat)d\t pass low statistics (or big deltax.error) check
  %(nmoved)d\t moved
Not moved counters:
  %(nnotmoved)d\t chambers not moved
  Numbers of chambers were not moved due to:
    %(nfail_toideal)d\t big shift with respect to ideal
    %(nfail_deltafinal)d\t big deltas during final iteration
    %(nfail_lowstat)d\t low statistics (or big deltax.error)
    %(nfail_nsigma)d\t |x_final - x_initial| < nsigma
""" % vars()

if DO_DT: loopover("DT")
if DO_CSC: loopover("CSC")

geom0.xml(file(theOutXML, "w"))
