#! /usr/bin/env python

import sys

execfile("plotscripts.py")

#ROOT.gROOT.SetBatch(1)

cargs = sys.argv[:]

if len(cargs) != 5 :
  print "Draws delta corrections from report1 (x axis) vs. from report2 (y axis)."
  print "usage: ./reportVsReport.py label selector report1.py report2.py"
  print "where selector is one of ALL, DT, CSC, CSCE1, CSCE2"
  print "The label will be included into the filename as reportVsReport_label_selector.png"
  sys.exit()


def DT(dt, wheel, station, sector): return dt == "DT"

def CSC(csc, endcap, station, ring, chamber): return csc == "CSC"

def CSCE1(csc, endcap, station, ring, chamber): return csc == "CSC" and endcap==1

def CSCE2(csc, endcap, station, ring, chamber): return csc == "CSC" and endcap==2


label = cargs[1]

selection = cargs[2]
if selection == 'ALL': selection = None

reportfile1 = cargs[3]
reportfile2 = cargs[4]
execfile(reportfile1)
r1 = reports
execfile(reportfile2)
r2 = reports

c1 = ROOT.TCanvas("c1","c1",760,800)

ranges = "window=25"
#if selection=="DT": ranges = "windows=[25.,5,1.,1.,5.,5.]"

eval("reportsDiff(r1, r2, %s, selection=%s, bins=2001, canvas=c1)" % (ranges, selection))
print "reportsDiff(r1, r2, %s, selection=%s, bins=2001)" % (ranges, selection)

c1.Update()
c1.Print("reportVsReport_%s_%s.png" % (label, cargs[2]))
