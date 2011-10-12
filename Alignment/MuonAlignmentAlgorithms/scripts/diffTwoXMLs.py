#! /usr/bin/env python

import sys, ROOT
ROOT.gROOT.SetBatch(1)

execfile("geometryXMLparser.py")
execfile("plotscripts.py")


cargs = sys.argv[:]

if len(cargs) < 5 or len(cargs) > 7:
  print "Draws differences between two xml geometries (or between single xml geometry and ideal) for all six variables."
  print "usage: ./diffTwoXMLs.py label selector geom2.xml geom1.xml report2.py report1.py"
  print "or     ./diffTwoXMLs.py label selector geom2.xml geom1.xml report.py"
  print "or     ./diffTwoXMLs.py label selector geom.xml report.py"
  print "where selector is one of ALL, DT, CSC, CSCE1, CSCE2"
  print "The label will be included into the filename as diffTwoXMLs_label_selector.png"
  print ""
  print "Special consistency test mode with respect to corrections in the report:"
  print "If the label starts with \"vsReport_\", the delta corrections from report2 would be substracted from the XML differences."
  print "Example:"
  print "./diffTwoXMLs.py diffReport_label selector geom2.xml geom1.xml report.py"
  sys.exit()


def ALL(dt, wheel, station, sector): return True

def ALL(csc, endcap, station, ring, chamber): return True

def DT(dt, wheel, station, sector): return dt == "DT"
def DT_st1(dt, wheel, station, sector): return dt == "DT" and station == 1
def DT_st2(dt, wheel, station, sector): return dt == "DT" and station == 2
def DT_st3(dt, wheel, station, sector): return dt == "DT" and station == 3
def DT_st4(dt, wheel, station, sector): return dt == "DT" and station == 4

def CSC(csc, endcap, station, ring, chamber): return csc == "CSC"
def CSCE1(csc, endcap, station, ring, chamber): return csc == "CSC" and endcap==1
def CSCE2(csc, endcap, station, ring, chamber): return csc == "CSC" and endcap==2


label = cargs[1]
diffReport = False
if label.find("vsReport_") == 0: diffReport = True

selection = cargs[2]

if len(cargs) == 5:
  xmlfile2 = cargs[3]
  reportfile2 = cargs[4]
  g1 = None
  r1 = None

if len(cargs) == 6:
  xmlfile2 = cargs[3]
  xmlfile1 = cargs[4]
  reportfile2 = cargs[5]
  g1 = MuonGeometry(xmlfile1)
  r1 = None

if len(cargs) == 7:
  xmlfile2 = cargs[3]
  xmlfile1 = cargs[4]
  reportfile2 = cargs[5]
  reportfile1 = cargs[6]
  g1 = MuonGeometry(xmlfile1)
  execfile(reportfile1)
  r1 = reports

g2 = MuonGeometry(xmlfile2)
execfile(reportfile2)
r2 = reports

c1 = ROOT.TCanvas("c1","c1",1000,600)

if not diffReport: 
  # normal XML diff mode
  ranges = "window=10"
  #if selection=="DT": ranges = "windows=[25.,5,1.,1.,5.,5.]"
  eval("DBdiff(g2, g1, r2, r1, %s, selection=%s, phi=False, bins=251)" % (ranges, selection))

else:
  # Special consistency test mode with respect to corrections in the report:
  ranges = "window=0.02"
  eval("DBdiff(g2, g1, r2, r1, %s, selection=%s, phi=False, bins=1001, reportdiff=True, inlog=True)" % (ranges, selection))


c1.Update()
c1.Print("diffTwoXMLs_%s_%s.png" % (label, selection))
