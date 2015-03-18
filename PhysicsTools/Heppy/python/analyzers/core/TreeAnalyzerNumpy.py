from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.tree import Tree as Tree
from ROOT import TFile

class TreeAnalyzerNumpy( Analyzer ):
    """Base TreeAnalyzerNumpy, to create flat TTrees.

    Check out TestTreeAnalyzer for a concrete example.
    IMPORTANT: FOR NOW, CANNOT RUN SEVERAL TreeAnalyzers AT THE SAME TIME!
    Anyway, you want only one TTree, don't you?"""

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(TreeAnalyzerNumpy,self).__init__(cfg_ana, cfg_comp, looperName)
        self.outservicename = getattr(cfg_ana,"outservicename","outputfile")
        self.treename = getattr(cfg_ana,"treename","tree")


    def beginLoop(self, setup) :
        super(TreeAnalyzerNumpy, self).beginLoop(setup)
        if self.outservicename in setup.services:
            print "Using outputfile given in", self.outservicename
            self.file = setup.services[self.outservicename].file
        else :
            fileName = '/'.join([self.dirName,
                             'tree.root'])
            isCompressed = self.cfg_ana.isCompressed if hasattr(self.cfg_ana,'isCompressed') else 1
            print 'Compression', isCompressed
            self.file = TFile( fileName, 'recreate', '', isCompressed )
        self.file.cd()
        if self.file.Get(self.treename) :
            raise RuntimeError, "You are booking two Trees with the same name in the same file"
        self.tree = Tree(self.treename, self.name)
        self.tree.setDefaultFloatType(getattr(self.cfg_ana, 'defaultFloatType','D')); # or 'F'
        self.declareVariables(setup)
        
    def declareVariables(self,setup):
        print 'TreeAnalyzerNumpy.declareVariables : overload this function.'
        pass

    def write(self, setup):
        super(TreeAnalyzerNumpy, self).write(setup)
        if self.outservicename not in setup.services:
            self.file.Write() 

