import glob

from ROOT import TChain, TFile, TTree

class Chain( object ):
    def __init__(self, treeName, pattern ):
        self.files = []
        if treeName is None:
            treeName = self.guessTreeName(pattern)
        self.chain = TChain(treeName)
        nFiles = 0
        for file in glob.glob(pattern):
            self.chain.Add(file)
            nFiles += 1
        if nFiles==0:
            raise ValueError('no matching file name: '+pattern)

    def guessTreeName(self, pattern):
        names = []
        for fnam in glob.glob(pattern):
            rfile = TFile(fnam)
            for key in rfile.GetListOfKeys():
                obj = rfile.Get(key.GetName())
                if type(obj) is TTree:
                    names.append( key.GetName() )
        thename = set(names)
        if len(thename)==1:
            return list(thename)[0]


    def __getattr__(self, attr):
        return getattr(self.chain, attr)

if __name__ == '__main__':

    import sys

    treeName = sys.argv[1]
    pattern = sys.argv[2]
    chain = Chain( treeName, pattern )
