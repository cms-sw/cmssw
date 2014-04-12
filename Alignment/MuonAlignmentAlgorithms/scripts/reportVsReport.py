#! /usr/bin/env python

import sys, optparse

copyargs = sys.argv[:]

for i in range(len(copyargs)):
  if copyargs[i] == "":             copyargs[i] = "\"\""
  if copyargs[i].find(" ") != -1:   copyargs[i] = "\"%s\"" % copyargs[i]
commandline = " ".join(copyargs)
print commandline

prog = sys.argv[0]

usage = """Usage:
%(prog)s [options] reportX.py reportY.py

Draws a scatterplot of delta corrections from reportX vs. from reportY.
""" % vars()

parser = optparse.OptionParser(usage)
parser.add_option("-o", "--output",
                  help="plots' file name. If not give, an automatic file name would be given as reportVsReport_label_selection.png",
                  type="string",
                  default="",
                  dest="filename")
parser.add_option("-l", "--label",
                  help="label for an automatic filename",
                  type="string",
                  default="",
                  dest="label")
parser.add_option("-s", "--selection",
                   help="is one of the following: ALL, DT, CSC, CSCE1, CSCE2",
                   type="string",
                   default="ALL",
                   dest="selection")
parser.add_option("-x", "--xlabel",
                   help="prefix to add to plots' X axis",
                   type="string",
                   default="None",
                   dest="xlabel")
parser.add_option("-y", "--ylabel",
                   help="prefix to add to plots' Y axis",
                   type="string",
                   default="None",
                   dest="ylabel")
parser.add_option("-w", "--which",
                   help="binary mask for which variables to draw, defauls is 110011",
                   type="string",
                   default="110011",
                   dest="which")
                   
options, args = parser.parse_args(sys.argv[1:])

if len(args)!=2:   print usage; sys.exit()


### definitions of selectors:

def DT(dt, wheel, station, sector): return dt == "DT"

def CSC(csc, endcap, station, ring, chamber): 
  if csc != "CSC": return False
  # skip the duplicated ME1/a
  if station==1 and ring==4: return False
  # skip non-instrumented ME4/2's:
  if station==4 and ring==2 and ( (endcap==1 and (chamber<9 or chamber >13)) or endcap==2 ) : return False
  return True

def CSCE1(csc, endcap, station, ring, chamber): return CSC(csc, endcap, station, ring, chamber) and endcap==1

def CSCE2(csc, endcap, station, ring, chamber): return CSC(csc, endcap, station, ring, chamber) and endcap==2


### main part

execfile("plotscripts.py")

ROOT.gROOT.SetBatch(1)

selection = options.selection
if selection == 'ALL': selection = None

execfile(args[0])
rx = reports
execfile(args[1])
ry = reports

if options.which.count('1')>4: c1 = ROOT.TCanvas("c1","c1",1000,800)
else: c1 = ROOT.TCanvas("c1","c1",760,800)

print "corrections2D(reportsX=rx, reportsY=ry, selection=%s, pre_title_x='%s', pre_title_y='%s', which='%s' )" % ( 
      selection, options.xlabel, options.ylabel, options.which )
eval( "corrections2D(reportsX=rx, reportsY=ry, selection=%s, pre_title_x='%s', pre_title_y='%s', which='%s', canvas=c1 )" % ( 
      selection, options.xlabel, options.ylabel, options.which) )

c1.Update()
if len(options.filename)>0: filename = options.filename
else: filename = "reportVsReport_"+options.label+"_"+options.selection+".png"
c1.Print(filename)
