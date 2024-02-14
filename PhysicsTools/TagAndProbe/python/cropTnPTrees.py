#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

import sys

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("inputFile", type=str)
parser.add_argument("fraction", type=float)
parser.add_argument("outputFile", type=str)
options = parser.parse_args()

frac = options.fraction
input  = ROOT.TFile(options.inputFile)
output = ROOT.TFile(options.outputFile, "RECREATE")
for k in input.GetListOfKeys():
    print(k.GetName(), k.GetClassName())
    if k.GetClassName() == "TDirectoryFile":
        print("  processing directory ",k.GetName())
        din  = input.Get(k.GetName())
        dout = output.mkdir(k.GetName())
        for i in din.GetListOfKeys():
            if i.GetClassName() == "TTree":
                src = din.Get(i.GetName()) #i.ReadObj(); # ReadObj doesn't work!!!
                newEntries = int(src.GetEntries()*frac)
                print("    cropped TTree",i.GetName(),", original entries",src.GetEntries(), ", new entries",newEntries)
                cloned = src.CloneTree(newEntries)
                dout.WriteTObject(cloned, i.GetName())       
            elif i.GetClassName() != "TDirectory":
                dout.WriteTObject(i.ReadObj(), i.GetName())
                print("    copied ",i.GetClassName(),i.GetName())


