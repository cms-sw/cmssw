# importing the python binding to the C++ class from ROOT 
from ROOT import heppy

class BTagSF(heppy.BTagSF):
    '''Just an additional wrapper, not really needed :-)
    We just want to illustrate the fact that you could
    use such a wrapper to add functions, attributes, etc,
    in an improved interface to the original C++ class. 
    '''
    def __init__ (self, seed) :
        super(BTagSF, self).__init__(seed) 

if __name__ == '__main__':

    btag = BTagSF(12345)
    print 'created BTagSF instance'
