#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *

## Compute the PDF uncertainties on an extrapolation
## defined as the ratio between the bin 3 and bin 2 
## of specified plot 

def replicaPdfRatio(file, base, pdf, eigenvectors):
    ## Compute the mean and RMS of the extrapolation
    ## over the replicas of a PDF 
    ## (to use e.g. with NNPDF)
    values  = [ ]
    for e in xrange(eigenvectors+1):
        hist = file.Get("%s_%s_%d" % (base,pdf,e))
        values.append( hist.GetBinContent(3)/hist.GetBinContent(2) )
    avg = sum([values[i] for i in xrange(1,eigenvectors+1)])/eigenvectors
    rms = sqrt(sum([(values[i]-avg)**2  for i in xrange(1,eigenvectors+1)])/(eigenvectors-1))
    return [ avg, avg-rms, avg+rms ]

def eigenPdfRatio(file, base, pdf, eigenvectors):
    ## Compute the central value and uncertainties 
    ## on a extrapolation using PDF eigenvectors
    ## (to use e.g. with CT10 and MSTW)
    if (eigenvectors % 2 != 0): raise RuntimeError
    central = file.Get("%s_%s_%d" % (base,pdf,0))
    cval = central.GetBinContent(3)/central.GetBinContent(2) 
    sumhi, sumlo = 0., 0. 
    for e in xrange(eigenvectors/2):
        h1 = file.Get("%s_%s_%d" % (base,pdf,2*e+1))
        h2 = file.Get("%s_%s_%d" % (base,pdf,2*e+2))
        d1 = h1.GetBinContent(3)/h1.GetBinContent(2) - cval
        d2 = h2.GetBinContent(3)/h2.GetBinContent(2) - cval
        dlo = min([0,d1,d2])
        dhi = max([0,d1,d2])
        sumlo += dlo**2
        sumhi += dhi**2
    return [ cval, cval - sqrt(sumlo), cval + sqrt(sumhi) ]
    

if __name__ == "__main__":
    from sys import argv
    if len(argv) != 4:
        print "Usage: %s plots.root plotName processName" % argv[0]
        exit(1)
    fin = ROOT.TFile(argv[1])
    #var = "wzControlRegions"; P = "WZ"
    var = argv[2]; P=argv[3]
    bandN = replicaPdfRatio(fin, var+"_"+P, "NNPDF21_100",    100)
    bandC = eigenPdfRatio(fin,   var+"_"+P, "CT10",           52)
    bandM = eigenPdfRatio(fin,   var+"_"+P, "MSTW2008lo68cl", 38)
    lhcband = [ bandC[0], min([bandN[1],bandC[1],bandM[1]]), max([bandN[2],bandC[2],bandM[2]]) ]
    print "Central value and band for ratio according to NNPDF: %.4f [%.4f, %.4f]" % (bandN[0],bandN[1],bandN[2])
    print "Central value and band for ratio according to CT10 : %.4f [%.4f, %.4f]" % (bandC[0],bandC[1],bandC[2])
    print "Central value and band for ratio according to MSTW : %.4f [%.4f, %.4f]" % (bandM[0],bandM[1],bandM[2])
    print "Central value and band for ratio, full envelope    : %.4f [%.4f, %.4f]" % (lhcband[0],lhcband[1],lhcband[2])
    print "Relative uncertainy: -%.1f/+%.1f %%" % (100.0*(lhcband[0]-lhcband[1])/lhcband[0], 100.0*(lhcband[2]-lhcband[0])/lhcband[0]) 

