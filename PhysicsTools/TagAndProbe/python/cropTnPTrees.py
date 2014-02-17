#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

import sys

from optparse import OptionParser
parser = OptionParser(usage = "usage: %prog [options] inputFile fraction outputFile",
                      version = "%prog $Id: cropTnPTrees.py,v 1.1 2010/05/07 14:22:37 gpetrucc Exp $")
(options, args) = parser.parse_args()

if len(args) <= 2: 
    parser.print_usage()
    sys.exit(2)
    
try:
    frac = float(args[1])
except TypeError:
    parser.print_usage()
    print "fraction must be a floating point number (e.g. 0.5)"
    sys.exit(2)

input  = ROOT.TFile(args[0])
output = ROOT.TFile(args[2], "RECREATE")
for k in input.GetListOfKeys():
    print k.GetName(), k.GetClassName()
    if k.GetClassName() == "TDirectoryFile":
        print "  processing directory ",k.GetName()
        din  = input.Get(k.GetName())
        dout = output.mkdir(k.GetName())
        for i in din.GetListOfKeys():
            if i.GetClassName() == "TTree":
                src = din.Get(i.GetName()) #i.ReadObj(); # ReadObj doesn't work!!!
                newEntries = int(src.GetEntries()*frac)
                print "    cropped TTree",i.GetName(),", original entries",src.GetEntries(), ", new entries",newEntries
                cloned = src.CloneTree(newEntries)
                dout.WriteTObject(cloned, i.GetName())       
            elif i.GetClassName() != "TDirectory":
                dout.WriteTObject(i.ReadObj(), i.GetName())
                print "    copied ",i.GetClassName(),i.GetName()


