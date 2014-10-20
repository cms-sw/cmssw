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
        fileName = '/'.join([self.dirName,
                             self.name+'_tree.root'])

        isCompressed = self.cfg_ana.isCompressed if hasattr(cfg_ana,'isCompressed') else 1
        print 'Compression', isCompressed

        self.file = TFile( fileName, 'recreate', '', isCompressed )
        self.tree = Tree(self.name,self.name)
        self.declareVariables()
        
    def declareVariables(self):
        print 'TreeAnalyzerNumpy.declareVariables : overload this function.'
        pass

    def write(self):
        super(TreeAnalyzerNumpy, self).write()
        self.file.Write() 

