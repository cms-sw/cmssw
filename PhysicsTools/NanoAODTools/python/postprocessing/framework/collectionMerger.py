from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
import ROOT
import numpy as np
import itertools
ROOT.PyConfig.IgnoreCommandLineOptions = True

_rootLeafType2rootBranchType = {
    'UChar_t': 'b',
    'Char_t': 'B',
    'UInt_t': 'i',
    'Int_t': 'I',
    'Float_t': 'F',
    'Double_t': 'D',
    'ULong64_t': 'l',
    'Long64_t': 'L',
    'Bool_t': 'O'
}


class collectionMerger(Module):
    def __init__(self,
                 input,
                 output,
                 sortkey=lambda x: x.pt,
                 reverse=True,
                 selector=None,
                 maxObjects=None):
        self.input = input
        self.output = output
        self.nInputs = len(self.input)
        self.sortkey = lambda obj_j_i1: sortkey(obj_j_i1[0])
        self.reverse = reverse
        # pass dict([(collection_name,lambda obj : selection(obj)])
        self.selector = [(selector[coll] if coll in selector else
                          (lambda x: True))
                         for coll in self.input] if selector else None
        # save only the first maxObjects objects passing the selection in the merged collection
        self.maxObjects = maxObjects
        self.branchType = {}
        pass

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):

        # Find list of activated branches in input tree
        _brlist_in = inputTree.GetListOfBranches()
        branches_in = set(
            [_brlist_in.At(i) for i in range(_brlist_in.GetEntries())])
        branches_in = [
            x for x in branches_in if inputTree.GetBranchStatus(x.GetName())
        ]

        # Find list of activated branches in output tree
        _brlist_out = wrappedOutputTree._tree.GetListOfBranches()
        branches_out = set(
            [_brlist_out.At(i) for i in range(_brlist_out.GetEntries())])
        branches_out = [
            x for x in branches_out
            if wrappedOutputTree._tree.GetBranchStatus(x.GetName())
        ]

        # Use both
        branches = branches_in + branches_out

        # Only keep branches with right collection name
        self.brlist_sep = [
            self.filterBranchNames(branches, x) for x in self.input
        ]
        self.brlist_all = set(itertools.chain(*(self.brlist_sep)))

        self.is_there = np.zeros(shape=(len(self.brlist_all), self.nInputs),
                                 dtype=bool)
        for bridx, br in enumerate(self.brlist_all):
            for j in range(self.nInputs):
                if br in self.brlist_sep[j]:
                    self.is_there[bridx][j] = True

        # Create output branches
        self.out = wrappedOutputTree
        for br in self.brlist_all:
            self.out.branch("%s_%s" % (self.output, br),
                            _rootLeafType2rootBranchType[self.branchType[br]],
                            lenVar="n%s" % self.output)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def filterBranchNames(self, branches, collection):
        out = []
        for br in branches:
            name = br.GetName()
            if not name.startswith(collection + '_'):
                continue
            out.append(name.replace(collection + '_', ''))
            self.branchType[out[-1]] = br.FindLeaf(br.GetName()).GetTypeName()
        return out

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        coll = [Collection(event, x) for x in self.input]
        objects = [(coll[j][i], j, i) for j in range(self.nInputs)
                   for i in range(len(coll[j]))]
        if self.selector:
            objects = [
                obj_j_i for obj_j_i in objects
                if self.selector[obj_j_i[1]](obj_j_i[0])
            ]
        objects.sort(key=self.sortkey, reverse=self.reverse)
        if self.maxObjects:
            objects = objects[:self.maxObjects]
        for bridx, br in enumerate(self.brlist_all):
            out = []
            for obj, j, i in objects:
                out.append(getattr(obj, br) if self.is_there[bridx][j] else 0)
            self.out.fillBranch("%s_%s" % (self.output, br), out)
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed

lepMerger = lambda: collectionMerger(input=["Electron", "Muon"],
                                     output="Lepton")
lepMerger_exampleSelection = lambda: collectionMerger(
    input=["Electron", "Muon"],
    output=
    "Lepton",  # this will keep only the two leading leptons among electrons with pt > 20 and muons with pt > 40
    maxObjects=2,
    selector=dict([("Electron", lambda x: x.pt > 20),
                   ("Muon", lambda x: x.pt > 40)]),
)
