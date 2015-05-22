"""
Load the libraries needed to use RootTools  
"""
# import ROOT as rt

from ROOT import gROOT, TFile, TCanvas, TPad, gPad, TBrowser, TH2F, TH1F, TH1D , TProfile, TLegend, gDirectory

def loadLibs():
    print 'loading FWLite.'
    #load the libaries needed
    from ROOT import gROOT,gSystem
    gSystem.Load("libFWCoreFWLite")
    gROOT.ProcessLine('AutoLibraryLoader::enable();')
    gSystem.Load("libFWCoreFWLite")
    gSystem.Load("libCintex")
    gROOT.ProcessLine('ROOT::Cintex::Cintex::Enable();')
        
    #now the RootTools stuff
    gSystem.Load("libCMGToolsRootTools")

loadLibs()



from CMGTools.RootTools.Chain import Chain
# from ROOT import Chain as CChain
# Chain = CChain
# Chain.__doc__ = """
# An extention of TChain so that it can take a glob in its constructor
# """

class stliter(object):
    """
Defines a python iterator for stl types    
    """
    def __init__(self, stl):
        self.stl = stl
        self.index = 0
    
    def __iter__(self):
        return self
    def next(self):
        if self.index < self.stl.size() - 1:
            self.index += 1
        else:
            raise StopIteration()
        return self.stl.at(self.index)
    
    def __len__(self):
        return self.stl.size()
