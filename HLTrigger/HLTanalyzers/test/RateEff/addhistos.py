import math,ROOT,sys,os
from ROOT import gROOT, TFile, TChain, TTree, TH1F, TF1,SetOwnership


if __name__ == '__main__':

    gROOT.Reset()

    outfile="hltmenu_8TeV_7.0e33_20130930_QCDRate.root"

    outf = TFile(outfile,"RECREATE");
    SetOwnership( outf, False )   # tell python not to take ownership
    print "Output written to: ", outfile

    narg=len(sys.argv)
    if narg < 2 :
        print 'Usage: python addhistos.py files*.root'
        sys.exit(1)

    rootfiles = []
    for ifile in range(1,narg):   
        rootfiles.append(sys.argv[ifile])

    for f in range(0, narg-1):
        
        print rootfiles[f]
        infile = TFile.Open(rootfiles[f])

        histInd = infile.Get("individual")
        histCum = infile.Get("cumulative")
        nevt = infile.Get("NEVTS")
        
        if f==0:
            histInd_all = histInd.Clone()
            histCum_all = histCum.Clone()
            NEVTS = nevt.Clone()
        else:
            histInd_all.Add(histInd)
            histCum_all.Add(histCum)
            NEVTS.Add(nevt)

        
    outf.cd()
    histInd_all.Write()
    histCum_all.Write()
    NEVTS.Write()
    outf.Close()
