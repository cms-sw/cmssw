# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import glob
import os
import pprint
from ROOT import TChain, TFile, TTree, gSystem

def is_pfn(fn):
    return not (is_lfn(fn) or is_rootfn(fn))

def is_lfn(fn):
    return fn.startswith("/store")

def is_rootfn(fn):
    """
    To open files like root://, file:// which os.isfile won't find.
    """
    return "://" in fn


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
        if isinstance(input, basestring): # input is a pattern
            self.files = glob.glob(input)
            if len(self.files)==0:
                raise ValueError('no matching file name: '+input)
        else: # case of a list of files
            if False in [
                ((is_pfn(fnam) and os.path.isfile(fnam)) or
                is_lfn(fnam)) or is_rootfn(fnam)
                for fnam in self.files]:
                err = 'at least one input file does not exist\n'
                err += pprint.pformat(self.files)
                raise ValueError(err)
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


