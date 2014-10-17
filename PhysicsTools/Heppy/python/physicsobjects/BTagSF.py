from CMGTools.RootTools.RootTools import loadLibs
from ROOT import gSystem

# loading the shared library containing the BTagSF class in ROOT
gSystem.Load("libCMGToolsRootTools")

# importing the python binding to the C++ class from ROOT 
from ROOT import BTagSF as BTagSF_CC

class BTagSF:
    '''Just an additional wrapper, not really needed :-)
    We just want to illustrate the fact that you could
    use such a wrapper to add functions, attributes, etc,
    in an improved interface to the original C++ class. 
    '''
    def __init__ (self, seed) :
        self.BTagSFcalc = BTagSF_CC(seed) 

if __name__ == '__main__':

    btag = BTagSF(12345)
    print 'created BTagSF instance'
