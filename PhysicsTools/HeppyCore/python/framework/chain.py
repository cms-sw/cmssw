# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import glob
from ROOT import TChain, TFile, TTree, gSystem

class Chain( object ):
    """Wrapper to TChain, with a python iterable interface.

    Example of use:  #TODO make that a doctest / nose?
       from chain import Chain
       the_chain = Chain('../test/test_*.root', 'test_tree')
       event3 = the_chain[2]
       print event3.var1

       for event in the_chain:
           print event.var1
    """

    def __init__(self, input, tree_name=None):
        """
        Create a chain.

        Parameters:
          input     = either a list of files or a wildcard (e.g. 'subdir/*.root').
                      In the latter case all files matching the pattern will be used
                      to build the chain.
          tree_name = key of the tree in each file.
                      if None and if each file contains only one TTree,
                      this TTree is used.
        """
        self.files = input
        if isinstance(input, basestring):
            self.files = glob.glob(input)
        if len(self.files)==0:
            raise ValueError('no matching file name: '+input)
        if tree_name is None:
            tree_name = self._guessTreeName(input)
        self.chain = TChain(tree_name)
        for file in self.files:
            self.chain.Add(file)

    def _guessTreeName(self, pattern):
        """
        Find the set of keys of all TTrees in all files matching pattern.
        If the set contains only one key
          Returns: the TTree key
        else raises ValueError.
        """
        names = []
        for fnam in self.files:
            rfile = TFile(fnam)
            for key in rfile.GetListOfKeys():
                obj = rfile.Get(key.GetName())
                if type(obj) is TTree:
                    names.append( key.GetName() )
        thename = set(names)
        if len(thename)==1:
            return list(thename)[0]
        else:
            err = [
                'several TTree keys in {pattern}:'.format(
                    pattern=pattern
                    ),
                ','.join(thename)
                ]
            raise ValueError('\n'.join(err))

    def __getattr__(self, attr):
        """
        All functions of the wrapped TChain are made available
        """
        return getattr(self.chain, attr)

    def __iter__(self):
        return iter(self.chain)

    def __len__(self):
        return int(self.chain.GetEntries())

    def __getitem__(self, index):
        """
        Returns the event at position index.
        """
        self.chain.GetEntry(index)
        return self.chain


if __name__ == '__main__':

    import sys

    if len(sys.argv)!=3:
        print 'usage: Chain.py <tree_name> <pattern>'
        sys.exit(1)
    tree_name = sys.argv[1]
    pattern = sys.argv[2]
    chain = Chain( tree_name, pattern )
