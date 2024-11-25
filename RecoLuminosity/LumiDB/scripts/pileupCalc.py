#!/usr/bin/env python3
from builtins import range
VERSION='1.00'
import os, sys, time
import argparse
from RecoLuminosity.LumiDB import pileupParser
from RecoLuminosity.LumiDB import selectionParser
import numpy as np
from scipy.special import loggamma

def parseInputFile(inputfilename):
    '''
    output ({run:[ls:[inlumi, meanint]]})
    '''
    selectf=open(inputfilename,'r')
    inputfilecontent=selectf.read()
    p=pileupParser.pileupParser(inputfilecontent)                            
    
#    p=inputFilesetParser.inputFilesetParser(inputfilename)
    runlsbyfile=p.runsandls()
    return runlsbyfile

def MyErf(xInput):
    # Abramowitz and Stegun approximations for Erf (equations 7.1.25-28)
    X = np.abs(xInput)

    p = 0.47047
    b1 = 0.3480242
    b2 = -0.0958798
    b3 = 0.7478556

    T = 1.0/(1.0+p*X)
    cErf = 1.0 - (b1*T + b2*T*T + b3*T*T*T)*np.exp(-1.0*X*X)

    # Alternate Erf approximation:
    #A1 = 0.278393
    #A2 = 0.230389
    #A3 = 0.000972
    #A4 = 0.078108

    #term = 1.0+ A1*X+ A2*X*X+ A3*X*X*X+ A4*X*X*X*X
    #denom = term*term*term*term

    #dErf = 1.0 - 1.0/denom

    return np.where(xInput < 0, -cErf, cErf)

def poisson(x, par):
    ## equivalent to TMath::Poisson (without x<0 and par<0 checks)
    return np.where(x == 0., np.exp(-par),
           np.exp(x*np.log(par)-loggamma(x+1)-par))

class EquidistantBinning(object):
    def __init__(self, num, xMin, xMax):
        self.num = num
        self.xMin = xMin
        self.xMax = xMax
        self.edges = np.linspace(xMin, xMax, num=num+1)
        self.centers = .5*(self.edges[:-1] + self.edges[1:])
    @property
    def width(self):
        return (self.xMax-self.xMin)/self.num
    def find(self, x):
        return np.floor((x-self.xMin)*self.num/(self.xMax-self.xMin)).astype(int)

Sqrt2 = np.sqrt(2)

def fillPileupHistogram(lumiInfo, calcOption, binning, hContents, minbXsec, run, ls):
    '''
    lumiinfo:[intlumi per LS, mean interactions ]

    intlumi is the deadtime corrected average integrated lumi per lumisection
    '''

    LSintLumi = lumiInfo[0]
    RMSInt = lumiInfo[1]*minbXsec
    AveNumInt = lumiInfo[2]*minbXsec

    # First, re-constitute lumi distribution for this LS from RMS:
    if RMSInt > 0:
        areaAbove = MyErf((AveNumInt-binning.edges)/Sqrt2/RMSInt)
        ## area above edge, so areaAbove[i]-areaAbove[i+1] = area in bin
        ProbFromRMS = .5*(areaAbove[:-1]-areaAbove[1:])
    else:
        ProbFromRMS = np.zeros(hContents.shape)
        obs = binning.find(AveNumInt)
        if ( obs < binning.num ) and ( AveNumInt >= 1.0E-5 ): # just ignore zero values
            ProbFromRMS[obs] = 1.0
        
    if calcOption == 'true':  # Just put distribution into histogram
        if RMSInt > 0:
            hContents += ProbFromRMS*LSintLumi
            totalProb = np.sum(ProbFromRMS)

            if 1.0-totalProb > 0.01:
                print("Run %d, LS %d: Significant probability density outside of your histogram (mean %.2f," % (run, ls, AveNumInt))
                print("rms %.2f, integrated probability %.3f). Consider using a higher value of --maxPileupBin." % (RMSInt, totalProb))
        else:
            hContents[obs] += LSintLumi ## obs = FindBin(AveNumInt), -1 because hContents has no overflows
    else: # have to convolute with a poisson distribution to get observed Nint
        if not hasattr(binning, "poissConv"): ## only depends on binning, cache
            ## poissConv[i,j] = TMath.Poisson(e[i], c[j])
            binning.poissConv = poisson(
                binning.edges[:-1,np.newaxis], ## e'[i,] = e[i]
                binning.centers[np.newaxis,:]) ## c'[,j] = c[j]
        # prob[i] = sum_j ProbFromRMS[j]*TMath.Poisson(e[i], c[j])
        prob = np.sum(binning.poissConv * ProbFromRMS[np.newaxis,:], axis=1)
        hContents += prob*LSintLumi
        #if ( not np.all(ProbFromRMS == 0) ) and 1.0-np.sum(prob) > 0.01:
        #    print("Run %d, LS %d: significant probability density outside of your histogram, %f" % (run, ls, np.sum(prob)))
        #    print("Consider using a higher value of --maxPileupBin")

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Script to estimate pileup distribution using bunch instantaneous luminosity information and minimum bias cross section. Outputs a TH1D of the pileup distribution stored in a ROOT file.")

    # required
    req_group = parser.add_argument_group('required arguments')
    req_group.add_argument('outputfile', action='store', help='output ROOT file')
    req_group.add_argument('-i', '--input', dest='inputfile', action='store', required=True,
                           help='input Run/LS file for your analysis in JSON format')
    req_group.add_argument('-j', '--inputLumiJSON', dest='inputLumiJSON', action='store', required=True,
                           help='input pileup file in JSON format')
    req_group.add_argument('-c', '--calcMode' ,dest='calcMode', action='store',
                           help='calculate either "true" or "observed" distributions',
                           choices=['true', 'observed'], required=True)

    # optional
    parser.add_argument('-x', '--minBiasXsec', dest='minBiasXsec', action='store',
                           type=float, default=69200.0,
                           help='minimum bias cross section to use (in microbarn) (default: %(default).0f)')
    parser.add_argument('-m', '--maxPileupBin', dest='maxPileupBin', action='store',
                           type=int, default=100, help='maximum value of pileup histogram (default: %(default)d)')
    parser.add_argument('-n', '--numPileupBins', dest='numPileupBins', action='store',
                           type=int, default=1000, help='number of bins in pileup histogram (default: %(default)d)')
    parser.add_argument('--pileupHistName', dest='pileupHistName', action='store',
                           default='pileup', help='name of pileup histogram (default: %(default)s)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                           help='verbose mode for printing' )

    options = parser.parse_args()
    output = options.outputfile
    
    if options.verbose:
        print('General configuration')
        print('\toutputfile:' ,options.outputfile)
        print('\tAction:' ,options.calcMode, 'luminosity distribution will be calculated')
        print('\tinput selection file:', options.inputfile)
        print('\tinput lumi JSON:', options.inputLumiJSON)
        print('\tMinBiasXsec:', options.minBiasXsec)
        print('\tmaxPileupBin:', options.maxPileupBin)
        print('\tnumPileupBins:', options.numPileupBins)

    binning = EquidistantBinning(options.numPileupBins, 0., options.maxPileupBin)
    hContents = np.zeros(binning.centers.shape)

    inpf = open(options.inputfile, 'r')
    inputfilecontent = inpf.read()
    inputRange = selectionParser.selectionParser (inputfilecontent).runsandls()

    #inputRange=inputFilesetParser.inputFilesetParser(options.inputfile)

    inputPileupRange=parseInputFile(options.inputLumiJSON)

    # now, we have to find the information for the input runs and lumi sections
    # in the Lumi/Pileup list. First, loop over inputs

    for (run, lslist) in sorted (inputRange.items()):
        # now, look for matching run, then match lumi sections
        # print "searching for run %d" % (run)
        if run in inputPileupRange.keys():
            #print run
            LSPUlist = inputPileupRange[run]
            # print "LSPUlist", LSPUlist
            for LSnumber in lslist:
                if LSnumber in LSPUlist.keys():
                    #print "found LS %d" % (LSnumber)
                    lumiInfo = LSPUlist[LSnumber]
                    # print lumiInfo
                    fillPileupHistogram(lumiInfo, options.calcMode,
                                        binning, hContents,
                                        options.minBiasXsec, run, LSnumber)
                else: # trouble
                    print("Run %d, LumiSection %d not found in Lumi/Pileup input file. Check your files!" \
                            % (run,LSnumber))

        else:  # trouble
            print("Run %d not found in Lumi/Pileup input file.  Check your files!" % (run))

        # print run
        # print lslist

    ## convert hContents to TH1F
    import ROOT
    pileupHist = ROOT.TH1D(options.pileupHistName, options.pileupHistName,
            options.numPileupBins, 0., options.maxPileupBin)
    for i,ct in enumerate(hContents):
        pileupHist.SetBinContent(i+1, ct)

    histFile = ROOT.TFile.Open(output, 'recreate')
    if not histFile:
        raise RuntimeError("Could not open '%s' as an output root file" % output)
    pileupHist.Write()
    histFile.Close()
    print("Wrote output histogram to", output)
