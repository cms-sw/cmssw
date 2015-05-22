#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *

def replicaPdfNorm(file, base, pdf, eigenvectors):
    vals = []
    for e in xrange(eigenvectors+1):
        vals.append( file.Get("%s_%s_%d" % (base,pdf,e)).Integral() )
    avg = sum([v for v in vals[1:]])/eigenvectors
    rms = sqrt(sum([(v-avg)**2 for v in vals[1:]])/eigenvectors)
    return [ avg - rms, vals[0], avg + rms ]

def eigenPdfNorm(file, base, pdf, eigenvectors):
    if (eigenvectors % 2 != 0): raise RuntimeError
    cen = file.Get("%s_%s_%d" % (base,pdf,0)).Integral()
    ret = [0, cen, 0]
    for e in xrange(eigenvectors/2):
        n1 = file.Get("%s_%s_%d" % (base,pdf,2*e+1)).Integral()
        n2 = file.Get("%s_%s_%d" % (base,pdf,2*e+2)).Integral()
        ddn = min(n2-cen,n1-cen,0)
        dup = max(n2-cen,n1-cen,0) 
        ret[0] += ddn**2 
        ret[2] += dup**2 
    ret[0] = cen - sqrt(ret[0])
    ret[2] = cen + sqrt(ret[2])
    return ret
    
if __name__ == "__main__":
    from sys import argv
    fin = ROOT.TFile(argv[1])
    var = "nJet25"
    for P in "ttH", "TTW", "TTZ":
        bandN = replicaPdfNorm(fin, var+"_"+P, "NNPDF21_100", 100)
        bandC = eigenPdfNorm(fin, var+"_"+P, "CT10", 52)
        bandM = eigenPdfNorm(fin, var+"_"+P, "MSTW2008lo68cl", 38)
        #print "NNPDF: ",bandN
        #print "CT10:  ",bandC
        #print "MSTW:  ",bandN
        cen  = bandC[1]
        emin = min(bandN[0], bandC[0], bandM[0])
        emax = max(bandN[2], bandC[2], bandM[2])
        print "uncertainties for %s: -%.1f/+%.1f %%, kappa = %.3f " % (P, 100.0*(1-emin/cen), 100.0*(emax/cen-1), sqrt(emax/emin) )

