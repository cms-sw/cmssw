# report on the lumisections and runs present in your LuminosityBlocks TTree
# import this file and do:
#  lumisProcessedForEachRun(histos,lumi)
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Types as CfgTypes

from ROOT import gDirectory, TLegend,gPad, TCanvas, TH1F, TH2F

import pprint

minRun = 165000
maxRun = 168000

class MyHistograms:
    def __init__(self):
        pass

#def lumiHisto(histograms, tree, run, minLum=0, maxLum=2000):
#    nbins = maxLum - minLum    
#    histograms.hlum = TH1F('hlum_%d' % run,'%d;lumi'% run, nbins , minLum, maxLum )
#    tree.Draw('lumi>>'+histograms.hlum.GetName(),'run==%d' % run)
#    gPad.Modified()
#    gPad.Update()

#def runHisto( histograms, tree, minRun=165000,maxRun = 168000):
#    nbins = maxRun - minRun
#    histograms.hrun = TH1F('hrun',';Run number', nbins , minRun, maxRun ) 
#    tree.Draw('run>>'+histograms.hrun.GetName())
#    gPad.Modified()
#    gPad.Update()

def lumiVsRunHisto(histograms, tree, minRun=165000,maxRun = 168000,minLum=0,maxLum=3000 ):
    nbinsLum = maxLum - minLum    
    nbinsRun = maxRun - minRun
    histograms.hlumVSrun = TH2F('hlumVSrun',';Run number;Lumi number',
                                nbinsRun , minRun, maxRun, nbinsLum, minLum, maxLum)
    tree.Draw('lumi:run>>'+histograms.hlumVSrun.GetName(),'','col')
    gPad.Modified()
    gPad.Update()

def analyzeRuns(histograms):
    histo = histograms.hlumVSrun
    px = histo.ProjectionX()
    lumis = []
    for bin in range(1, histo.GetNbinsX() ):
        if px.GetBinContent(bin)==0:
            continue
        run = px.GetBinLowEdge(bin)
        histograms.runHisto = histo.ProjectionY("",bin,bin,"")
        histograms.runHisto.SetTitle('run %d' % run)
        histograms.runHisto.Draw()
        gPad.Modified()
        gPad.Update()
        tmp = lumisProcessed(histograms.runHisto, run)
        lumis.extend(tmp)
    pprint.pprint(lumis)
 
def lumisProcessed(histo, run):
    first = -1
    last = -1
    min = -1
    max = -1
    lumis = []
    for bin in range(1, histo.GetNbinsX() ):
        nEntries = histo.GetBinContent(bin)
        index = histo.GetBinLowEdge(bin)
        if nEntries:
            if min<0:
                min = index
        else:
            if min>-1:
                max = index-1
                # print min, max
                lumis.append( '%d:%d-%d:%d' % (run, min, run, max) )
                min = -1
                max = -1
    return lumis


def runsProcessed(histograms, tree):
    runHisto(histograms, tree)
    histo = histograms.hrun
    runs = []
    for bin in range(1, histo.GetNbinsX() ):
        nEntries = histo.GetBinContent(bin)
        run = histo.GetBinLowEdge(bin)
        if nEntries:
            runs.append( int(run) )
    return runs

def lumisProcessedForEachRun(histograms, tree):
    lumiVsRunHisto(histograms, tree)
    analyzeRuns(histograms)

histos = MyHistograms()

