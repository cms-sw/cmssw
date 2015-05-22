from CMGTools.RootTools import RootTools as tools
import ROOT as rt

class cmgTuple(rt.TObject):

    def branchAlias(self, branchName):
        names = branchName.split('_')

        result = branchName
        if len(names) == 4:
            result = names[1]
        return result

    def __init__(self, tree):

        rt.TObject.__init__(self)

        self.tree = tree
        self.currentEntry = 0

        s = set()
        for i in xrange(self.tree.GetListOfBranches().GetEntries()):
            b = self.tree.GetListOfBranches().At(i)
            name = b.GetName()
            s.add(name)

        self.aliases = {}
        self.branches = []

        for b in s:
            name = self.branchAlias(b)
            if name != b and name not in self.aliases:
                #set a branch alias for the individual objects
                self.aliases[name] = b
                self.tree.SetAlias(name,'%sobj' % b)

                #and also the collections
                self.tree.SetAlias('%sVec' % name,'%s@obj' % b)
            self.branches.append(b)

#        print "Created cmgTuple for TTree called '%s' containing " \
#              "the following branches and aliases:" % \
#              self.tree.GetName()
#        for (i, j) in self.aliases.iteritems():
#            print "  %s -> %s" % (i, j)

    def __len__(self):
        return self.tree.GetEntries()

    def __iter__(self):
        return self

    def __del__(self):
        if self.tree is not None and self.tree:
            self.tree.Delete()

    def next(self):
        if(self.currentEntry < len(self)):
            self.tree.GetEntry(self.currentEntry)
            self.currentEntry += 1
        else:
            raise StopIteration()
        return self

    def printAliases( self ):
        keys = self.aliases.keys()
        lengths = [len(i) for i in keys]
        max_len = max(lengths)
        keys.sort()
        for key in keys:
            alias = self.aliases[key]
            print "  %-*s -> %s" % (max_len, key, alias)

    def get(self, name):
        result = None
        if name in self.aliases:
            result = getattr(self.tree, self.aliases[name])
        elif name in self.branches:
            result = getattr(self.tree, name)
        elif '%s.' % name in self.branches:
            result = getattr(self.tree,'%s.' % name)
        else:
            raise NameError("'%s' is not a branch in the TTree" % name)
        return result
    def Draw(self, *args):
        return self.tree.Draw(*args)
    def Scan(self, *args):
        return self.tree.Scan(*args)
    def Print(self, *args):
        return self.tree.Print(*args)

if __name__ == '__main__':

    import sys
    input = rt.TFile.Open(sys.argv[1])
    events = input.Get('Events')

    cmg = cmgTuple(events)
    print cmg.branches
    print cmg.aliases
