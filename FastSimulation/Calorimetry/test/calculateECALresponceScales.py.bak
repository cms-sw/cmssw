#!/usr/bin/env python

import ConfigParser
import argparse
import numpy

import ROOT

parser = argparse.ArgumentParser(description="Calculate the scale factors used to correct the energy response of the ECAL in FastSim.")
parser.add_argument("--cfg", help="Name of the configuration file.", default="calculateECALresponceScales.cfg")
args = parser.parse_args()

config = ConfigParser.SafeConfigParser()
config.readfp( open(args.cfg) )

# build 3d histogram
title = ";"+";".join( [config.get("xAxis", "title"),config.get("yAxis", "title"),config.get("zAxis", "title")] )

binEdges = {}
for axis in "xAxis", "yAxis", "zAxis":
    if config.has_option(axis, "binCenters"):
        binCenters = [ float(i) for i in config.get(axis, "binCenters").split(",") ]
        # To find the correct scale, TH3::Interpolate is used, which can not use underflow and overflow.
        # Therefore, minimum and maximum energy has to be defined.

        binEdgeList = [0]
        for i in range(len(binCenters)-1):
            binEdgeList.append( 0.5*(binCenters[i]+binCenters[i+1] ) )
        binEdgeList.append( max([1e4, 10*binEdgeList[-1]]) ) # maximum energy is arbritrary

        binEdges[axis] = numpy.array( binEdgeList )
    else:
        binEdges[axis] = numpy.linspace(
            config.getfloat(axis, "min"),
            config.getfloat(axis, "max"),
            num=config.getint(axis,"bins") )


h3 = ROOT.TH3F(
    config.get("output", "histogramName"),
    title,
    len( binEdges["xAxis"] )-1, binEdges["xAxis"],
    len( binEdges["yAxis"] )-1, binEdges["yAxis"],
    len( binEdges["zAxis"] )-1, binEdges["zAxis"]
  )


ROOT.gROOT.LoadMacro("calculateECALresponceScales.C+")

fac = ROOT.KKFactorsFactory( h3,
        config.get("input", "fileNameFast"),
        config.get("input", "fileNameFull"),
        config.get("input", "treeName")
      )

fac.calculate()

fout = ROOT.TFile( config.get("output", "fileName"), "recreate" )
fac.GetH3().Write()
fout.Close()


