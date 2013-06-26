#!/usr/bin/env python

# J.P. Chou, Brown University
# February 20, 2009

# import libraries
from ROOT import gROOT, TCanvas, TH1F, gPad, gStyle, TFile
import re, os, sys

# parse command line: look for a log file
if len(sys.argv) < 2:
    print "This is a script to parse a log file for HcalNoiseInfoProducer timing information"
    print "usage:", sys.argv[0], "logfile [rootoutfile]"
    sys.exit(1)
logfile = sys.argv[1]
rootfile = ""
if len(sys.argv) >= 3:
    rootfile = sys.argv[2]

# load root stuff
# a user should modify this if there is a problem
gROOT.Reset()
gROOT.Macro( os.path.expanduser( '~/root/defaultrootlogon.C' ) )

# make canvas and histogram
canvas = TCanvas('c1', 'c1',800,400)
canvas.Divide(2,1)
hist = TH1F('timing','Time to Run HcalNoiseInfoProducer',100,0.0,1000.)
hist.GetXaxis().SetTitle('time [ms]')

# create regular expression
p = re.compile('hcalnoiseinfoproducer HcalNoiseInfoProducer')

# parse file
in_file = open(logfile,"r")
while True:
    in_line = in_file.readline()
    if not in_line:
        break
    if p.search(in_line):
        x = 1000*float(in_line.split()[5])
        hist.Fill(x)

# draw stuff
gStyle.SetOptStat(111110)
canvas.cd(1)
hist.Draw()
canvas.cd(2)
hist.Draw()
gPad.SetLogy(1)
canvas.Update()

# if we have a root file, write the histogram
if rootfile:
    print "Writing histogram to root file: ", rootfile
    f=TFile.Open(rootfile, 'RECREATE')
    hist.Write()
    f.Close()

raw_input("Press Enter to Continue:")
