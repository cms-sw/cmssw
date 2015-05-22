import sys
import copy
import pprint 
from CMGTools.RootTools.PyRoot import * 
from ROOT import TFile, TH1F

fileNames = sys.argv[1:]
hists = []
files = []


def getHisto(fileName):
    files.append( TFile(fileName) ) 
    pu = files[-1].Get('pileup')
    pu.Scale( 1/pu.Integral() )
    hists.append( pu )


def weight( hs, weights ):
    start = True
    outh = None
    totWeight = 0
    # import pdb; pdb.set_trace()
    for h, weight in zip(hs, weights):
        if start:
            outh = copy.deepcopy(h)
            outh.Reset()
            start = False
        outh.Add(h, weight)
        totWeight += weight
    outh.Scale(1/totWeight)
    return outh
        

for fname in fileNames:
    getHisto(fname)

