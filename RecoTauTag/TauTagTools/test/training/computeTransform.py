#!/usr/bin/env python

'''

Build a cms.PSet desribing a "Tanc transformation".  The TaNC transformation
f(x) transforms the neural network output x of a tau candidate such that f(x) is
the probability that it is a true tau.

The transformation "t" format is a cms.vdouble, where the ith entry is the value
of f(x) at x = i*(max-min)/(len(t)-1) + min

Authors: Evan K. Friis, Christian Veelken (UC Davis)

'''

from RecoLuminosity.LumiDB import argparse

parser = argparse.ArgumentParser(
    description = "Build a python module containing object 'transform', which"
    " contains the transform paramterization."
)
parser.add_argument('-s', metavar='file', help='Signal transform file')
parser.add_argument('-b', metavar='file', help='Background transform file')
parser.add_argument('-o',  help='Output file')

options=parser.parse_args()

import FWCore.ParameterSet.Config as cms
import ROOT
import sys

print "Building transformation:", options.o

ROOT.gROOT.SetBatch(True)

signal_file = ROOT.TFile(options.s, "READ")
background_file = ROOT.TFile(options.b, "READ")

signal_denominator_histo = signal_file.Get("plotInputJets/pt")
background_denominator_histo = background_file.Get("plotInputJets/pt")
signal_histo = signal_file.Get("cleanTauPlots/hpsTancTausDiscriminationByTancRaw")
background_histo = background_file.Get("cleanTauPlots/hpsTancTausDiscriminationByTancRaw")

print "Signal has %i entries in clean, %i in total" % (
    signal_histo.Integral(), signal_denominator_histo.Integral())
print "Background has %i entries in clean, %i in total" % (
    background_histo.Integral(), background_denominator_histo.Integral())

# Determine the probability for a given jet to end up in this decay mode stream
signal_scale = signal_histo.Integral()/signal_denominator_histo.Integral()
background_scale = \
        background_histo.Integral()/background_denominator_histo.Integral()

signal_integral = signal_histo.GetIntegral()
background_integral = background_histo.GetIntegral()

min = signal_histo.GetBinCenter(1)
max = signal_histo.GetBinCenter(signal_histo.GetNbinsX())

transform_values = []

last_value = -1
for bin in range(1, signal_histo.GetNbinsX()+1):
    bin_center = signal_histo.GetBinCenter(bin)
    signal_passing = 1.0 - signal_integral[bin]
    background_passing = 1.0 - background_integral[bin]
    transform_value = last_value
    if signal_passing > 0:
        transform_value = ((signal_passing*signal_scale) /
                           (signal_passing*signal_scale +
                            background_passing*background_scale))
        last_value = transform_value
    transform_values.append(transform_value)

output_object = cms.PSet(
    min = cms.double(min),
    max = cms.double(max),
    transform = cms.vdouble(transform_values)
)
output_file = open(options.o, 'w')
output_file.write('import FWCore.ParameterSet.Config as cms\n')
output_file.write("transform = %s\n" % output_object.dumpPython())

output_file.close()

print "Transform file %s created" % options.o

sys.exit(0)
