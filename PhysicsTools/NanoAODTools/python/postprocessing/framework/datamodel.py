from PhysicsTools.NanoAODTools.postprocessing.framework.treeReaderArrayTools import InputTree
import ROOT
import math
ROOT.PyConfig.IgnoreCommandLineOptions = True

statusflags = { # GenPart_statusFlags, stored bitwise (powers of 2):
  # https://github.com/cms-sw/cmssw/edit/master/PhysicsTools/NanoAOD/python/genparticles_cff.py
  # https://cms-nanoaod-integration.web.cern.ch/integration/master-106X/mc106Xul18_doc.html#GenPart
  'isPrompt':                      (1 << 0),   'fromHardProcess':                    (1 <<  8),
  'isDecayedLeptonHadron':         (1 << 1),   'isHardProcessTauDecayProduct':       (1 <<  9),
  'isTauDecayProduct':             (1 << 2),   'isDirectHardProcessTauDecayProduct': (1 << 10),
  'isPromptTauDecayProduct':       (1 << 3),   'fromHardProcessBeforeFSR':           (1 << 11),
  'isDirectTauDecayProduct':       (1 << 4),   'isFirstCopy':                        (1 << 12),
  'isDirectPromptTauDecayProduct': (1 << 5),   'isLastCopy':                         (1 << 13),
  'isDirectHadronDecayProduct':    (1 << 6),   'isLastCopyBeforeFSR':                (1 << 14),
  'isHardProcess':                 (1 << 7),
}


class Event:
    """Class that allows seeing an entry of a PyROOT TTree as an Event"""

    def __init__(self, tree, entry):
        self._tree = tree
        self._entry = entry
        self._tree.gotoEntry(entry)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return self._tree.readBranch(name)

    def __getitem__(self, attr):
        return self.__getattr__(attr)

    def eval(self, expr):
        """Evaluate an expression, as TTree::Draw would do. 

           This is added for convenience, but it may perform poorly and the implementation is not bulletproof,
           so it's better to rely on reading values, collections or objects directly
        """
        if not hasattr(self._tree, '_exprs'):
            self._tree._exprs = {}
            # remove useless warning about EvalInstance()
            import warnings
            warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                    message='creating converter for unknown type "const char\*\*"$')
            warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                    message='creating converter for unknown type "const char\*\[\]"$')
        if expr not in self._tree._exprs:
            formula = ROOT.TTreeFormula(expr, expr, self._tree)
            if formula.IsInteger():
                formula.go = formula.EvalInstance64
            else:
                formula.go = formula.EvalInstance
            self._tree._exprs[expr] = formula
            # force sync, to be safe
            self._tree.GetEntry(self._entry)
            self._tree.entry = self._entry
            # self._tree._exprs[expr].SetQuickLoad(False)
        else:
            self._tree.gotoEntry(self._entry)
            formula = self._tree._exprs[expr]
        if "[" in expr:  # unclear why this is needed, but otherwise for some arrays x[i] == 0 for all i > 0
            formula.GetNdata()
        return formula.go()


class Object:
    """Class that allows seeing a set branches plus possibly an index as an Object"""

    def __init__(self, event, prefix, index=None):
        self._event = event
        self._prefix = prefix + "_"
        self._index = index

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError
        val = getattr(self._event, self._prefix + name)
        if self._index != None:
            val = val[self._index]
        # convert char to integer number
        val = ord(val) if type(val) == str else val
        self.__dict__[name] = val  # cache
        return val

    def __getitem__(self, attr):
        return self.__getattr__(attr)

    def p4(self, corr_pt=None):
        """Create TLorentzVector for this particle."""
        ret = ROOT.TLorentzVector()
        if corr_pt == None:
            ret.SetPtEtaPhiM(self.pt, self.eta, self.phi, self.mass)
        else:
            ret.SetPtEtaPhiM(corr_pt, self.eta, self.phi, self.mass)
        return ret

    def DeltaR(self, other):
        if isinstance(other, ROOT.TLorentzVector):
            deta = abs(other.Eta() - self.eta)
            dphi = abs(other.Phi() - self.phi)
        else:
            deta = abs(other.eta - self.eta)
            dphi = abs(other.phi - self.phi)
        while dphi > math.pi:
            dphi = abs(dphi - 2 * math.pi)
        return math.sqrt(dphi**2 + deta**2)

    def statusflag(self, flag):
        """Find if bit for statusflag is set (for GenPart only)."""
        return (self.statusFlags & statusflags[flag])==statusflags[flag]

    def subObj(self, prefix):
        return Object(self._event, self._prefix + prefix)

    def __repr__(self):
        return ("<%s[%s]>" % (self._prefix[:-1], self._index)) if self._index != None else ("<%s>" % self._prefix[:-1])

    def __str__(self):
        return self.__repr__()


class Collection:
    def __init__(self, event, prefix, lenVar=None):
        self._event = event
        self._prefix = prefix
        if lenVar != None:
            self._len = getattr(event, lenVar)
        else:
            self._len = getattr(event, "n" + prefix)
        self._cache = {}

    def __getitem__(self, index):
        if type(index) == int and index in self._cache:
            return self._cache[index]
        if index >= self._len:
            raise IndexError("Invalid index %r (len is %r) at %s" % (index, self._len, self._prefix))
        elif index < 0:
            raise IndexError("Invalid index %r (negative) at %s" % (index, self._prefix))
        ret = Object(self._event, self._prefix, index=index)
        if type(index) == int:
            self._cache[index] = ret
        return ret

    def __len__(self):
        return self._len
