import os
import sys
import fnmatch
import ROOT

import CMGTools.RootTools.StartUp 

from CMGTools.RootTools.RootTools import *

gROOT.Macro( os.path.expanduser( '~/rootlogon.C' ) )

# adding current directory in PYTHONPATH
sys.path.append('.')

from CMGTools.RootTools.Style import *


canvases = []
cx = 500
cy = 500

histTypes = [ ROOT.TH1F,
              ROOT.TH1D,
              ROOT.TH2F,
              ROOT.TH2D,
              ROOT.TProfile ]

dirTypes = [ROOT.TDirectory, ROOT.TDirectoryFile]

hists = None
trees = None
dirs = None

class Directory(object):
    '''Like TDirectory, but autoloads directory contents'''
    def __init__(self, tdir):
        self.tdir = tdir
    def __getattr__(self, attr):
        return getattr(self.tdir, attr)
    def cd(self, theLOCALS=None):
        if theLOCALS is None:
            theLOCALS = LOCALS
        self.tdir.cd()
        hists = load(theLOCALS, self, histTypes)
        trees = load(theLOCALS, self, [ROOT.TTree])
        dirs = load(theLOCALS, self, dirTypes)

def draw(pattern):
    '''Draw all histograms with a key matching pattern in separate canvases'''
    for hist in hists:
        name = hist.GetName()
        if fnmatch.fnmatch(name, pattern):
            can =  TCanvas(name, name, cx, cy)
            canvases.append( can )
            can.Draw()
            hist.Draw()

def load(locals, dir, types):
    '''Load objects in dir with a type in types.'''
    objs = []
    for key in dir.GetListOfKeys():
        obj = dir.Get(key.GetName())
        if type(obj) in types:
            print type(obj), key.GetName()
            if type(obj) in dirTypes:
                obj = Directory(obj)
            objs.append(obj)
            locals[key.GetName()] = obj
    return objs


def treeComp(t1, t2, var, cut):
    t1.SetWeight(1./t1.GetEntries(), 'global')
    t2.SetWeight(1./t2.GetEntries(), 'global')
    t1.Draw(var, cut)
    t2.Draw(var, cut, 'same')

if __name__ == '__main__':

    import sys
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.usage = '''
    PyRoot.py [root_file]
    '''
    
    (options,args) = parser.parse_args()

    file = None
    if len(args)>1:
        parser.print_help()
        print
        print 'Maximum one argument (a root file)'
        sys.exit(1)
    elif len(args)==1:
        file = TFile( args[0] )
        import ROOT
      
        LOCALS = locals()
        
        hists = load(LOCALS, file, histTypes)
        trees = load(LOCALS, file, [ROOT.TTree])
        dirs = load(LOCALS, file, dirTypes)

    
