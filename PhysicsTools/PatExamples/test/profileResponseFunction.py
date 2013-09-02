#!/usr/bin/env python

import sys, os, math, array, ROOT, getopt
from array import *

def main():

	print "\n \n \t  good morning! :-)  This script profiles the 2D response functions. \n \n  Usage: \n python "+sys.argv[0]+" -i inputfileWith2DHisto.root -o myOutputPlotName -d directoryOf2DHisto -n HistogramName \n \n"

	#Read in Parameters
	letters = 'i:o:d:n'
	keywords = ['input', 'output', 'dir', 'name']
	opts, extraparams = getopt.getopt(sys.argv[1:], letters, keywords)
	input='TFileServiceOutput.root'
	output='profiledResponseFunction'
	dir='jecAnalyzer'
        histname='Response'
	for o,p in opts:
		if o in ['-i','--input']:
			input = p
		elif o in ['-o','--output']:
			output = p
		elif o in ['-d','--dir']:
			dir = p
                elif o in ['-n','--name']:
                        histname = p

	print 'Input:',input
	print 'Output: ',output
        print 'Directory: ', dir
        print 'Histogram name: ', histname
        
	# ReadIn File
 	inputfile = ROOT.TFile.Open(input)
        # Get 2D Histo
        histo2d = inputfile.Get("%s/%s" % (dir, histname))
        # Profiling
        c = ROOT.TCanvas("canvas", "canvas", 800, 800)
        c.cd()
        prof = histo2d.ProfileX()
        prof.GetYaxis().SetTitle("p_{T}(reco)/p_{T}(gen)")
        prof.SetTitle("Response function of "+dir)
        prof.Draw()
        c.Print(output+"_"+dir+"_"+histname+".png")

if __name__ == '__main__':
	data = main()
