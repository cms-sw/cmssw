import os
import fnmatch
import copy

from ROOT import TFile, TH1F, TPaveText

from CMGTools.RootTools.PyRoot import * 
from CMGTools.RootTools.utils.file_dir import file_dir
from CMGTools.RootTools.DataMC.DataMCPlot import DataMCPlot
from CMGTools.RootTools.Style import *
from CMGTools.RootTools.Style import *


class H2TauTauDataMCDC( DataMCPlot ):

    keeper = {}
    HINDEX = 0

    def __init__(self, fileName, hists):

        super(H2TauTauDataMCDC, self).__init__(fileName)
        self.legendBorders = 0.651, 0.463, 0.895, 0.892
        self._InitPrefs()
        self.tfile, self.tdir = file_dir( fileName )
        for name in hists:
            hist = self.tdir.Get(name)
            if hist == None:
                print 'skipping', name
                continue
            print 'adding', name
            # import pdb; pdb.set_trace()
            self.AddHistogram(name, hist, 0, name)
            if name.find('data')!=-1:
                self.Hist(name).stack = False
            self._ApplyPref( self.Hist(name) )


    def _ApplyPref(self, hist):
        for prefpat, pref in self.histPref.iteritems():
            if fnmatch.fnmatch( hist.name, prefpat ):
                hist.SetStyle( pref['style'] )
                hist.layer = pref['layer']
            
        
    def _InitPrefs(self):
        '''Definine preferences for each component'''
        self.histPref = {}
        self.histPref['data_obs'] = {'style':sBlack, 'layer':2002}
        self.histPref['ZTT'] = {'style':sHTT_DYJets, 'layer':4}
        self.histPref['ZL'] = {'style':sGreen, 'layer':3.2}
        self.histPref['ZJ'] = {'style':sBlue, 'layer':3.1}
        self.histPref['QCD'] = {'style':sHTT_QCD, 'layer':2}
        self.histPref['W'] = {'style':sHTT_WJets, 'layer':3}  
        self.histPref['TT'] = {'style':sHTT_TTJets, 'layer':1} 
        self.histPref['*qqH*'] = {'style':sHTT_Higgs, 'layer':1001}
        self.histPref['*ggH*'] = {'style':sHTT_Higgs, 'layer':1001}
        self.histPref['*VH*'] = {'style':sHTT_Higgs, 'layer':1001}


if __name__=='__main__':
    
    import sys
    htt = H2TauTauDataMCDC(sys.argv[1], ['data_obs','ZTT','ZL','ZJ','W','QCD','TT',
                                         'ggH120',
                                         'ggH125',
                                         'ggH130',
                                         'ggH135',
                                         'ggH140',
                                         'ggH145',
                                         'qqH120',
                                         'qqH125',
                                         'qqH130',
                                         'qqH135',
                                         'qqH140',
                                         'qqH145',
                                         'VH120',
                                         'VH125',
                                         'VH130', 
                                         'VH135', 
                                         'VH140',
                                         'VH145'] ) 

    # c = TCanvas()
    htt.DrawStack('HIST')
