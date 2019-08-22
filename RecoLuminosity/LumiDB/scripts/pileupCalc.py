#!/usr/bin/env python
from __future__ import print_function
from builtins import range
VERSION='1.00'
import os, sys, time
import argparse
from RecoLuminosity.LumiDB import pileupParser
from RecoLuminosity.LumiDB import selectionParser
from math import exp
from math import sqrt
import six

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

def MyErf(input):

    # Abramowitz and Stegun approximations for Erf (equations 7.1.25-28)
    X = abs(input)

    p = 0.47047
    b1 = 0.3480242
    b2 = -0.0958798
    b3 = 0.7478556

    T = 1.0/(1.0+p*X)
    cErf = 1.0 - (b1*T + b2*T*T + b3*T*T*T)*exp(-1.0*X*X)
    if input<0:
        cErf = -1.0*cErf

    # Alternate Erf approximation:
    
    #A1 = 0.278393
    #A2 = 0.230389
    #A3 = 0.000972
    #A4 = 0.078108

    #term = 1.0+ A1*X+ A2*X*X+ A3*X*X*X+ A4*X*X*X*X
    #denom = term*term*term*term

    #dErf = 1.0 - 1.0/denom
    #if input<0:
    #    dErf = -1.0*dErf
        
    return cErf


def fillPileupHistogram (lumiInfo, calcOption, hist, minbXsec, Nbins, run, ls):
    '''
    lumiinfo:[intlumi per LS, mean interactions ]

    intlumi is the deadtime corrected average integrated lumi per lumisection
    '''

    LSintLumi = lumiInfo[0]
    RMSInt = lumiInfo[1]*minbXsec
    AveNumInt = lumiInfo[2]*minbXsec

    #coeff = 0

    #if RMSInt > 0:
    #    coeff = 1.0/RMSInt/sqrt(6.283185)

    #expon = 2.0*RMSInt*RMSInt

    Sqrt2 = sqrt(2)

    ##Nbins = hist.GetXaxis().GetNbins()

    ProbFromRMS = []
    BinWidth = hist.GetBinWidth(1)

    # First, re-constitute lumi distribution for this LS from RMS:
    if RMSInt > 0:

        AreaLnew = -10.
        AreaL = 0

        for obs in range (Nbins):
            #Old Gaussian normalization; inaccurate for small rms and large bins
            #val = hist.GetBinCenter(obs+1)
            #prob = coeff*exp(-1.0*(val-AveNumInt)*(val-AveNumInt)/expon)
            #ProbFromRMS.append(prob)
            
            left = hist.GetBinLowEdge(obs+1)
            right = left+BinWidth

            argR = (AveNumInt-right)/Sqrt2/RMSInt
            AreaR = MyErf(argR)

            if AreaLnew<-5.:
                argL = (AveNumInt-left)/Sqrt2/RMSInt
                AreaL = MyErf(argL)
            else:
                AreaL = AreaLnew
                AreaLnew = AreaR  # save R bin value for L next time

            NewProb = (AreaL-AreaR)*0.5

            ProbFromRMS.append(NewProb)

            #print left, right, argL, argR, AreaL, AreaR, NewProb

    else:
        obs = hist.FindBin(AveNumInt)
        for bin in range (Nbins):
            ProbFromRMS.append(0.0)
        if obs<Nbins+1:            
            ProbFromRMS[obs] = 1.0
        if AveNumInt < 1.0E-5:
           ProbFromRMS[obs] = 0.  # just ignore zero values
        
    if calcOption == 'true':  # Just put distribution into histogram
        if RMSInt > 0:
            totalProb = 0
            for obs in range (Nbins):
                prob = ProbFromRMS[obs]
                val = hist.GetBinCenter(obs+1)
                #print obs, val, RMSInt,coeff,expon,prob
                totalProb += prob
                hist.Fill (val, prob * LSintLumi)
                
            if 1.0-totalProb > 0.01:
                print("Run %d, LS %d: Significant probability density outside of your histogram (mean %.2f," % (run, ls, AveNumInt))
                print("rms %.2f, integrated probability %.3f). Consider using a higher value of --maxPileupBin." % (RMSInt, totalProb))
        else:
            hist.Fill(AveNumInt,LSintLumi)
    else: # have to convolute with a poisson distribution to get observed Nint
        totalProb = 0
        Peak = 0
        BinWidth = hist.GetBinWidth(1)
        for obs in range (Nbins):
            Peak = hist.GetBinCenter(obs+1)
            RMSWeight = ProbFromRMS[obs]
            for bin in range (Nbins):
                val = hist.GetBinCenter(bin+1)-0.5*BinWidth
                prob = ROOT.TMath.Poisson (val, Peak)
                totalProb += prob
                hist.Fill (val, prob * LSintLumi * RMSWeight)

        if 1.0-totalProb > 0.01:
            print("Run %d, LS %d: significant probability density outside of your histogram" % (run, ls))
            print("Consider using a higher value of --maxPileupBin")


    return hist



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

    import ROOT 
    pileupHist = ROOT.TH1D (options.pileupHistName,options.pileupHistName,
                      options.numPileupBins,
                      0., options.maxPileupBin)

    nbins = options.numPileupBins
    upper = options.maxPileupBin

    inpf = open(options.inputfile, 'r')
    inputfilecontent = inpf.read()
    inputRange = selectionParser.selectionParser (inputfilecontent).runsandls()

    #inputRange=inputFilesetParser.inputFilesetParser(options.inputfile)

    inputPileupRange=parseInputFile(options.inputLumiJSON)

    # now, we have to find the information for the input runs and lumi sections
    # in the Lumi/Pileup list. First, loop over inputs

    for (run, lslist) in sorted (six.iteritems(inputRange)):
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
                    fillPileupHistogram(lumiInfo, options.calcMode, pileupHist,
                                        options.minBiasXsec, nbins, run, LSnumber)
                else: # trouble
                    print("Run %d, LumiSection %d not found in Lumi/Pileup input file. Check your files!" \
                            % (run,LSnumber))

        else:  # trouble
            print("Run %d not found in Lumi/Pileup input file.  Check your files!" % (run))

        # print run
        # print lslist

    histFile = ROOT.TFile.Open(output, 'recreate')
    if not histFile:
        raise RuntimeError("Could not open '%s' as an output root file" % output)
    pileupHist.Write()
    #for hist in histList:
    #    hist.Write()
    histFile.Close()
    print("Wrote output histogram to", output)
