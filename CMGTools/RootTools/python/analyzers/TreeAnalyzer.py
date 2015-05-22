from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.statistics.Tree import Tree
from ROOT import TFile

class TreeAnalyzer( Analyzer ):
    """Base TreeAnalyzer, to create flat TTrees.

    Check out TestTreeAnalyzer for a concrete example.
    IMPORTANT: FOR NOW, CANNOT RUN SEVERAL TreeAnalyzers AT THE SAME TIME!
    Anyway, you want only one TTree, don't you?"""

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(TreeAnalyzer,self).__init__(cfg_ana, cfg_comp, looperName)
        fileName = '/'.join([self.dirName,
                             self.name+'_tree.root'])
        self.file = TFile( fileName, 'recreate' )
        self.tree = Tree(self.name,self.name)
        self.declareVariables()
        
    def declareVariables(self):
        print 'TreeAnalyzer.declareVariables : overload this function.'
        # self.tree.addVar('float', 'vismass')
        # self.tree.book()
        pass

    def write(self):
        super(TreeAnalyzer, self).write()
        self.file.Write() 

