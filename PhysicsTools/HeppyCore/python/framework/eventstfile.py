# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

from ROOT import TFile

class Events(object):
    '''Event list from a tree in a root file.
    '''
    def __init__(self, filename, treename, options=None):
        self.filename = filename
        self.treename = treename
        self.file = TFile(filename)
        if self.file.IsZombie():
            raise ValueError('file {fnam} does not exist'.format(fnam=filename))
        self.tree = self.file.Get(treename)
        if self.tree == None: # is None would not work
            raise ValueError('tree {tree} does not exist in file {fnam}'.format(
                tree = treename,
                fnam = filename
                ))

    def size(self):
        return self.tree.GetEntries()

    def to(self, iEv):
        '''navigate to event iEv.'''
        nbytes = self.tree.GetEntry(iEv)
        if nbytes < 0:
            raise IOError("Could not read event {0} in tree {1}:{2}".format(
                iEv, self.filename, self.treename
            ))
        return self.tree

    def __iter__(self):
        return iter(self.tree)
