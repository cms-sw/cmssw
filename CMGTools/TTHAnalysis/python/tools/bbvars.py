from CMGTools.TTHAnalysis.treeReAnalyzer import *

class bbVars:
    def __init__(self):
        self.branches = [ "minDrbbLoose", "minMbbLoose", "minDrbbMedium", "minMbbMedium" ]
        self.branches += [ "JetPt_%d" % i for i in xrange(1,9) ]
    def listBranches(self):
        return self.branches[:]
    def jjvals(self, jets, func):
        ret = []
        for i,j1 in enumerate(jets):
            for j2 in jets[i+1:]:
                ret.append(func(j1,j2))
        return ret  
    def __call__(self,event):
        jets    = [j for j in Collection(event,"Jet","nJet25",8)]
        bloose  = [j for j in jets if j.btagCSV > 0.244]
        bmedium = [j for j in jets if j.btagCSV > 0.679]
        ret = dict([(name,0.0) for name in self.branches])
        drbb = lambda j1, j2: deltaR(j1,j2)
        mbb  = lambda j1, j2: (j1.p4()+j2.p4()).M()
        if len(bloose) >= 2:
            ret["minDrbbLoose"] = min( self.jjvals(bloose, drbb) )
            ret["minMbbLoose"]  = min( self.jjvals(bloose, mbb ) )
        if len(bmedium) >= 2:
            ret["minDrbbMedium"] = min( self.jjvals(bmedium, drbb) )
            ret["minMbbMedium"]  = min( self.jjvals(bmedium, mbb ) )
        jpts = [ j.pt for j in jets]
        jpts.sort(); jpts.reverse()
        for i in xrange(1,9):
            ret["JetPt_%d" % i] = jpts[i-1] if i <= len(jets) else 0
        return ret
if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = bbVars()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: bloose %d, bmedium %d" % (ev.run, ev.lumi, ev.evt, ev.nBJetLoose25, ev.nBJetMedium25)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
