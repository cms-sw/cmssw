from ROOT import TH2F,gPad, TCanvas

import time

from CMGTools.RootTools.TreeFunctions import * 

def varStringToFileName( istr ):
    ostr = istr.replace( ' ', '_' )
    ostr = istr.replace( '.', 'o' )
    ostr = ostr.replace( '(', 'd' )
    ostr = ostr.replace( ')', 'b' )
    ostr = ostr.replace( '[', 'D' )
    ostr = ostr.replace( ']', 'B' )
    ostr = ostr.replace( '&&', 'AND' )
    ostr = ostr.replace( '||', 'OR' )
    ostr = ostr.replace( '>', 'GT' )
    ostr = ostr.replace( '<', 'LT' )
    return ostr

class CutCorrelations:
    def __init__(self, name):
        self.cuts = []
        self.name = name
        self.histo = None
        self.addtlCut = None
    def reset(self):
        self.cuts = []
        self.histo = None
        self.addtlCut = None
    def addCut( self, cut ):
        self.cuts.append( cut )
    def printCuts(self):
        for k, v in self.cuts:
            print k, '-->', v
    def bookCorrelationHisto( self ):
        nbins = len(self.cuts)
        hname = 'correlations_'+ self.name
        if self.addtlCut != None:
            hname += ' : ' + self.addtlCut
        self.histo = TH2F( hname, hname, nbins, 0, nbins, nbins, 0, nbins)
        self.histo.SetStats(0)
        i=0
        for key, v in self.cuts:
            i = i+1
            self.histo.GetYaxis().SetBinLabel(i, key)
            self.histo.GetXaxis().SetBinLabel(i, key)
        # self.histo.Draw()
        # time.sleep(4)
    def fillCorrelationHisto( self, tree, addtlCut='1'):
        if addtlCut != '1':
            setEventList( tree, addtlCut )
            self.addtlCut = addtlCut
        self.bookCorrelationHisto()
        i1 = 0
        for key1, v1 in self.cuts:
            i1 = i1+1
            i2 = 0
            for key2, v2 in self.cuts:
                i2 = i2+1
                if i2<i1: continue # the matrix is symmetric
                cut = v1 + ' && ' + v2
                print i1, i2
                n = tree.Draw('1', cut , 'goff')
                self.histo.SetBinContent(i1, i2, n)
                self.histo.SetBinContent(i2, i1, n)
        if addtlCut != '1':
            tree.SetEventList(0)
    def draw( self ):
        self.canvas = TCanvas('can'+self.name, self.name, 1000,800)
        if self.histo == None:
            self.bookCorrelationHisto()
        self.histo.Draw('boxtext')
        gPad.SetBottomMargin(0.3)
        gPad.SetLeftMargin(0.3)
        gPad.SetRightMargin(0.2)
        gPad.SaveAs('cutCorr_'+self.name+'_'+varStringToFileName(self.addtlCut)+'.png')
