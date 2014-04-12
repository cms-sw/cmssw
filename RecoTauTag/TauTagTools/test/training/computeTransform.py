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

min = signal_histo.GetBinCenter(1)
max = signal_histo.GetBinCenter(signal_histo.GetNbinsX())

output_object = cms.PSet(
    min = cms.double(min),
    max = cms.double(max),
)

transform_values = []

# Get the cut for a signal efficiencies of 60%, 40%, 20% (loose, medium, tight)
thresholds = [0.7, 0.6, 0.5]
threshold_values = []

def make_cdf(histogram):
    # Build the cumulative distribution function from a histogram.  We don't add
    # points on the graph where the CDF isn't changing (and we aren't on the
    # starting or ending plateu).  This is due to bad
    # statistics.
    points = []
    integral = histogram.GetIntegral()
    for ibin in range(0, histogram.GetNbinsX()+1):
        bin_center = histogram.GetBinCenter(ibin+1)
        cdf = integral[ibin]
        # Check if we have a new highest point to add, or are on a plateu
        if not points or cdf < 1e-6 or (1-cdf) < 1e-6 or cdf > points[-1][1]:
            points.append((bin_center, cdf))
    output = ROOT.TGraph(len(points))
    for i, (x, y) in enumerate(points):
        output.SetPoint(i, x, y)
    return output

def compute_transform(raw_cut, signal_cdf, signal_scale,
                      background_cdf, bkg_scale):
    """Compute the regularization transformation function"""

    signal_passing = (1.0 - signal_cdf.Eval(raw_cut))
    background_passing = (1.0 - background_cdf.Eval(raw_cut))

    signal_passing_weighted = signal_passing*signal_scale
    background_passing_weighted = background_passing*background_scale

    denominator = signal_passing_weighted + background_passing_weighted
    output = {
        'raw_cut' : raw_cut,
        'transform' : denominator and (signal_passing_weighted/denominator) or 0,
        'signal_passing' : signal_passing,
        'background_passing' : background_passing,
        'signal_passing_weighted' : signal_passing_weighted,
        'background_passing_weighted' : background_passing_weighted,
    }
    return output

# Build the cumulative distribution functions
print "Building signal c.d.f"
signal_cdf = make_cdf(signal_histo)
print "Building background c.d.f"
background_cdf = make_cdf(background_histo)
transform = lambda x: compute_transform(x, signal_cdf, signal_scale,
                                        background_cdf, background_scale)
npoints = 1000
for ix in range(npoints):
    x = min + ix*(max-min)*1.0/npoints
    transform_result = transform(x)
    transform_values.append(transform_result['transform'])
    # Check if this is one of our cuts
    if thresholds and transform_result['signal_passing'] < thresholds[0]:
        print "***********"
        print x, transform_result
        thresholds.pop(0)
        threshold_values.append((x, transform_result['transform']))

output_object.transform = cms.vdouble(transform_values)
# Store information about the weight of this decay mode for signal
output_object.signalDecayModeWeight = cms.double(signal_scale)
output_object.backgroundDecayModeWeight = cms.double(background_scale)

print threshold_values

# On the chance that a decay mode has nothing in it
if not threshold_values:
    threshold_values = [(-1,-1)]*3
# Store the loose medium and tight cut thresholds
output_object.looseCutRaw = cms.double(threshold_values[0][0])
output_object.looseCut = cms.double(threshold_values[0][1])
output_object.mediumCutRaw = cms.double(threshold_values[1][0])
output_object.mediumCut = cms.double(threshold_values[1][1])
output_object.tightCutRaw = cms.double(threshold_values[2][0])
output_object.tightCut = cms.double(threshold_values[2][1])

output_file = open(options.o, 'w')
output_file.write('import FWCore.ParameterSet.Config as cms\n')
output_file.write("transform = %s\n" % output_object.dumpPython())

output_file.close()

print "Transform file %s created" % options.o

sys.exit(0)
