import json
import re
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


class JSONFilter:
    def __init__(self, fname="", runsAndLumis={}):
        self.keep = {}
        if fname != "":
            self.runsAndLumis = json.load(open(fname, 'r'))
        else:
            self.runsAndLumis = runsAndLumis
        for _run, lumis in self.runsAndLumis.items():
            run = int(_run)
            if run not in self.keep:
                self.keep[run] = []
            self.keep[run] += lumis
        for run in list(self.keep.keys()):
            if len(self.keep[run]) == 0:
                del self.keep[run]

    def filterRunLumi(self, run, lumi):
        try:
            for (l1, l2) in self.keep[run]:
                if l1 <= lumi and lumi <= l2:
                    return True
            return False
        except KeyError:
            return False

    def filterRunOnly(self, run):
        return (run in self.keep)

    def runCut(self):
        return "%d <= run && run <= %s" % (min(self.keep.keys()), max(self.keep.keys()))

    def filterEList(self, tree, elist):
        # FIXME this can be optimized for sure
        tree.SetBranchStatus("*", 0)
        tree.SetBranchStatus('run', 1)
        tree.SetBranchStatus('luminosityBlock', 1)
        filteredList = ROOT.TEntryList('filteredList', 'filteredList')
        if elist:
            for i in range(elist.GetN()):
                entry = elist.GetEntry(0) if i == 0 else elist.Next()
                tree.GetEntry(entry)
                if self.filterRunLumi(tree.run, tree.luminosityBlock):
                    filteredList.Enter(entry)
        else:
            for entry in range(tree.GetEntries()):
                tree.GetEntry(entry)
                if self.filterRunLumi(tree.run, tree.luminosityBlock):
                    filteredList.Enter(entry)
        tree.SetBranchStatus("*", 1)
        return filteredList


def preSkim(tree, jsonInput=None, cutstring=None, maxEntries=None, firstEntry=0):
    if jsonInput == None and cutstring == None:
        return None, None
    cut = None
    jsonFilter = None
    if jsonInput != None:
        if type(jsonInput) is dict:
            jsonFilter = JSONFilter(runsAndLumis=jsonInput)
        else:
            jsonFilter = JSONFilter(jsonInput)
        cut = jsonFilter.runCut()
    if cutstring != None:
        cut = "(%s) && (%s)" % (cutstring, cut) if cut else cutstring
    if maxEntries is None:
        maxEntries = ROOT.TVirtualTreePlayer.kMaxEntries
    while "AltBranch$" in cut:
        m = re.search(r"AltBranch\$\(\s*(\w+)\s*,\s*(\w+)\s*\)", cut)
        if not m:
            raise RuntimeError(
                "Error, found AltBranch$ in cut string, but it doesn't comply with the syntax this code can support. The cut is %r" % cut)
        cut = cut.replace(m.group(0), m.group(
            1) if tree.GetBranch(m.group(1)) else m.group(2))
    tree.Draw('>>elist', cut, "entrylist", maxEntries, firstEntry)
    elist = ROOT.gDirectory.Get('elist')
    if jsonInput:
        elist = jsonFilter.filterEList(tree, elist)
    return elist, jsonFilter
